import numpy as np

import theano
from theano import tensor

#from blocks import initialization
from blocks.bricks import Identity, Linear, Tanh, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, BaseRecurrent
#from blocks.bricks.parallel import Merge
#from blocks.bricks.parallel import Fork

from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
#from blocks.roles import INPUT, WEIGHT, OUTPUT

from blocks.model import Model
from blocks.algorithms import (GradientDescent, Scale, StepClipping, CompositeRule)
from blocks.main_loop import MainLoop

from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.utils import named_copy

floatX = theano.config.floatX = 'float32'
# theano.config.assert_no_cpu_op='raise'

theano.config.compute_test_value = 'raise'
#theano.config.optimizer='None'  # Not a Python None
theano.config.optimizer='fast_compile'

import hickle
#word2vec = hickle.load('/home/andrewsm/SEER/services/deepner/server/data/embedding.0.hickle')
word2vec = hickle.load('/home/andrewsm/SEER/services/deepner/server/data/embedding.20.hickle')
embedding = word2vec['embedding']
code2word = word2vec['vocab']

last_element = len(code2word)-1
unk = embedding.mean(axis=0)
embedding[last_element] = unk
code2word[last_element] = '<UNK>'

word2code = {  v:i for i,v in enumerate(code2word) }

## These are set from the contents of the embedding file itself
#vocab_size=4
#embedding_dim=40

print("Embedding shape :", embedding.shape)                             # (4347, 100)
print("Embedding dtype :", embedding.dtype)                             # float32
vocab_size, embedding_dim = embedding.shape

extra_size = 1 # (caps,)

hidden_dim = embedding_dim+extra_size  # RNN units align over total embedding size

labels_size=5 # ( 0 .. 4 inclusive ), see range of CoNLLTextFile.ner values

max_sentence_length = 128 # There's a long sentence in testb... (124 words)
mini_batch_size = 8
# This becomes the size of the RNN 'output', 
# each place with a (hidden_dim*2) vector (x2 because it's bidirectional)

num_batches=10000  # For the whole main_loop

#batch_of_sentences = (mini_batch_size, max_sentence_length)
batch_of_sentences = (max_sentence_length, mini_batch_size)  # Since the data_stream has a _transpose

checkpoint_save_path='.'

"""
Deep BiRNN for Blocks

https://gist.github.com/rizar/183620f9cfec98f2acd4

This has additional classes that together can build a deep, bidirectional encoder
"""

"""
stack of LSTM

https://github.com/mila-udem/blocks/pull/688  :: Accepted! (Code is in 'blocks' core repo)

Update blocks from git in env 
pip install git+git://github.com/mila-udem/blocks.git@master
   -- suggests it needs '--upgrade' : Meh
   -- Direct approach with uninstall first (works without the git+git knobs):
         pip uninstall blocks
         # Fortunately, this doesn't touch the blocks-extras code (?)
         pip install git+git://github.com/mila-udem/blocks.git@master
   -- Alternative is to clone separately, and do :
         python setup.py install  
         # or
         python setup.py develop
         
Usage of RecurrentStack :
https://github.com/sotelo/poet/blob/master/poet.py
         
"""
import codecs
import re

from picklable_itertools import iter_, chain

from fuel.datasets import Dataset
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme

def _filter_long(data):
    return len(data[0]) <= max_sentence_length

def _transpose(data):
    #return tuple(array.T for array in data)  # Works for 1 and 2-d data
    #return tuple(np.transpose(array, axes=[1,0]) for array in data)  # No - need to specify all axes in this function
    return tuple(np.rollaxis(array, 1, 0) for array in data)  # Keeps tensor pieces sacrosanct too

class CoNLLTextFile(Dataset):
  provides_sources = ("tokens", "extras", "labels", )
  ner={
    'O'     :(0, 0), 
    'I-PER' :(0, 1), 
    'I-LOC' :(0, 2), 
    'I-ORG' :(0, 3), 
    'I-MISC':(0, 4), 
    'B-PER' :(1, 1), 
    'B-LOC' :(1, 2), 
    'B-ORG' :(1, 3), 
    'B-MISC':(1, 4), 
  }
  _digits = re.compile('\d')
  unknown = None

  def __init__(self, files, dictionary, unknown_token, **kwargs):
    self.files = files
    self.dictionary = dictionary 
    self.unknown = dictionary[unknown_token] # Error if not there
    
    super(CoNLLTextFile, self).__init__(**kwargs)

  def open(self):
    return chain(*[iter_( codecs.open(f, encoding="latin1") ) for f in self.files])
    #return codecs.open(self.fname, encoding="latin1")
      
  def get_data(self, state, request=None):
    if request is not None:
      raise ValueError
    tokens, extras, labels = [], [], []
    while True:
      # 'state' is the open file, read entries until we hit a ''
      line = next(state).rstrip()
      if len(line)==0:
        break
      if ' ' in line:
        l = line.split(' ')
        labels.append(self.ner[ l[-1] ][1] ) # Use just the second entry as the output target label...
        word = l[0]
      else: 
        word = line
      
      token=word.lower()
      caps = 0. if word == token else 1.
      
      if bool(self._digits.search(token)):
        #print("NUMBER found in %s" % (token))
        token = re.sub(r'\d+', 'NUMBER', token)
      
      tokens.append( self.dictionary.get(token, self.unknown) )
      
      spelling_ner = []
      extras.append( [ caps, ] + spelling_ner )  # include spelling-related NER vector
    
    if len(tokens)>0:
      return (np.array(tokens, dtype="int32"), np.array(extras, dtype=floatX), np.array(labels, dtype="int32"))   
    else:
      raise StopIteration

#data_paths = ['/home/andrewsm/SEER/external/CoNLL2003/ner/eng.testb',]  # 748Kb file
data_paths = ['/home/andrewsm/SEER/external/CoNLL2003/ner/eng.train',]  # 3.3Mb file
dataset = CoNLLTextFile(data_paths, dictionary=word2code, unknown_token='<UNK>')

data_stream = DataStream(dataset)
data_stream = Filter(data_stream, _filter_long)
#data_stream = Mapping(data_stream, reverse_words, add_sources=("targets",))

data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(mini_batch_size))

#data_stream = Padding(data_stream, mask_sources=('tokens'))            # Adds a mask fields to this stream field, type='floatX'
data_stream = Padding(data_stream, )              # Adds a mask fields to this stream field, type='floatX'
data_stream = Mapping(data_stream, _transpose)    # Flips stream so that sentences run down columns, batches along rows (strangely)

if False: # print sample
  #t=0
  max_len=0
  for i, data in enumerate(data_stream.get_epoch_iterator()):
    #print(i)
    #t=t + data[4].sum() + data[0].shape[1]
    l=data[0].shape[0]
    if l>max_len:
      max_len = l
    print(i, data[0].shape, max_len)
    #print(data)
    #break
  exit(0)
  
"""
Comments indicate that a reshaping has to be done, so let's think 
about sizes of the arrays...
"""

x = tensor.matrix('tokens', dtype="int32")

x_mask = tensor.matrix('tokens_mask', dtype=floatX)
#rnn.apply(inputs=input_to_hidden.apply(x), mask=x_mask)

lookup = LookupTable(vocab_size, embedding_dim)
#?? lookup.print_shapes=True

x_extra = tensor.tensor3('extras', dtype=floatX)

rnn = Bidirectional(
  SimpleRecurrent(dim=hidden_dim, activation=Tanh(),
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0),
  ),
)

### Will need to reshape the rnn outputs to produce suitable input here...
gather = Linear(name='hidden_to_output', 
  input_dim=hidden_dim*2, output_dim=labels_size,
  weights_init=IsotropicGaussian(0.01),
  biases_init=Constant(0)
)

p_labels = Softmax()



## Let's initialize the variables
lookup.allocate()
#print("lookup.parameters=", lookup.parameters)                         # ('lookup.parameters=', [W])

#lookup.weights_init = FUNCTION
#lookup.initialize() 

#lookup.params[0].set_value( np.random.normal( scale = 0.1, size=(vocab_size, embedding_dim) ).astype(np.float32) )
#lookup.params[0].set_value( embedding )

# See : https://github.com/mila-udem/blocks/blob/master/tests/bricks/test_lookup.py
#lookup.W.set_value(numpy.arange(15).reshape(5, 3).astype(theano.config.floatX))
lookup.W.set_value( embedding.astype(theano.config.floatX) )
    
rnn.initialize()
gather.initialize()



## Now for the application of these units

# Define the shape of x specifically...  :: the data has format (batch, features).
x.tag.test_value       = np.random.randint(vocab_size, size=batch_of_sentences ).astype(np.int32)
x_extra.tag.test_value = np.zeros( (max_sentence_length, mini_batch_size, 1) ).astype(np.float32)
x_mask.tag.test_value  = np.random.choice( [0.0, 1.0], size=batch_of_sentences ).astype(np.float32)

#tensor.reshape(x, batch_of_sentences  )
#x = tensor.specify_shape(x_base, batch_of_sentences)
#print("x (new) shape", x.shape.eval())
print("x shape", x.shape.tag.test_value)                                # array([29, 16]))

word_embedding = lookup.apply(x)
print("word_embedding shape", word_embedding.shape.tag.test_value)      # array([ 29, 16, 100]))
print("x_extra shape", x_extra.shape.tag.test_value)                    # array([ 29, 16,   1]))

embedding_extended = tensor.concatenate([ word_embedding, x_extra ], axis=-1)
print("embedding_extended shape", embedding_extended.shape.tag.test_value)   # array([ 29, 16, 101]))

rnn_outputs = rnn.apply(embedding_extended, mask=x_mask)
print("rnn_outputs shape", rnn_outputs.shape.tag.test_value)            # array([ 29, 16, 202]))

### So : Need to reshape the rnn outputs to produce suitable input here...
# Convert a tensor here into a long stream of vectors

#rnn_outputs_reshaped = rnn_outputs.reshape( (max_sentence_length*mini_batch_size, hidden_dim*2) )
rnn_outputs_reshaped = rnn_outputs.reshape( (x.shape[0]*x.shape[1], hidden_dim*2) )  # This depends on the batch... (and last one in epoch may be smaller)
print("rnn_outputs_reshaped shape", rnn_outputs_reshaped.shape.tag.test_value)   #array([464, 202]))

labels_raw = gather.apply(rnn_outputs_reshaped)  # This is pre-softmaxing
print("labels_raw shape", labels_raw.shape.tag.test_value)              # array([ 464, 5]))

label_probs = p_labels.apply(labels_raw)               # This is a list of label probabilities
print("label_probs shape", label_probs.shape.tag.test_value)            # array([ 464, 5]))            
# -- so :: this is an in-place rescaling

y = tensor.matrix('labels', dtype="int32")   # This is a symbolic vector of ints (implies one-hot in categorical_crossentropy)
y.tag.test_value = np.random.randint( labels_size, size=batch_of_sentences).astype(np.int32)

print("y shape", y.shape.tag.test_value)                                # array([ 29, 16]))
print("y.flatten() shape", y.flatten().shape.tag.test_value)            # array([464]))
print("y.flatten() dtype", y.flatten().dtype)                           # int32

"""
class CategoricalCrossEntropy(Cost):
    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        cost = tensor.nnet.categorical_crossentropy(y_hat, y).mean()
        return cost
"""
#cost = CategoricalCrossEntropy().apply(y.flatten(), label_probs)

## Version with mask : 
#cce = tensor.nnet.categorical_crossentropy(label_probs, y.flatten())
cce = tensor.nnet.crossentropy_categorical_1hot(label_probs, y.flatten())
y_mask = x_mask.flatten()
print("y_mask shape", y_mask.shape.tag.test_value)                      # array([464]))
print("y_mask dtype", y_mask.dtype)                                     # float32

cost = (cce * y_mask).sum() / y_mask.sum()             # elementwise multiple, followed by scaling 
cost.name='crossentropy_categorical_1hot_masked'

#cost = label_probs.sum()  # FAKE!
print("Created explicit cost:");
print(cost)

## Less explicit version
#mlp = MLP([Softmax()], [hidden_dim, labels_size],
#          weights_init=IsotropicGaussian(0.01),
#          biases_init=Constant(0))

#print(encoder.prototype.apply.sequences)
#dir(encoder.prototype.apply.sequences)

#probs = mlp.apply(encoder.apply(lookup.apply(x)))
#cost = CategoricalCrossEntropy().apply(y.flatten(), probs)

# Define the training algorithm.
cg = ComputationGraph(cost)

#print("Created ComputationGraph, variables:");
#print(cg.variables)

print("Created ComputationGraph, parameters:");
#print(cg.parameters)
for p in cg.parameters:
    print(str(p), p.shape, p.dtype)

print("Created ComputationGraph, inputs:");
print(cg.inputs)

algorithm = GradientDescent(
  cost=cost, 
  parameters=cg.parameters,
  step_rule=CompositeRule( [StepClipping(10.0), Scale(0.01), ] ),
)
print("Defined Algorithm");

model = Model(cost)
print("Defined Model");

obs_max_length = named_copy(x.shape[0], "obs_max_length")
observables = [
  cost, 
  obs_max_length,
  #min_energy, max_energy, 
  #mean_activation,
]

main_loop = MainLoop(
  model=model,
  data_stream=data_stream,
  algorithm=algorithm,
  extensions=[
    Timing(),
    TrainingDataMonitoring(observables, after_batch=True),
    #average_monitoring,
    #FinishAfter(after_n_batches=num_batches),
    FinishAfter(after_n_epochs=50),
    
    # Saving the model and the log separately is convenient,
    # because loading the whole pickle takes quite some time.
    Checkpoint(checkpoint_save_path, every_n_batches=500, save_separately=["model", "log"]),
    Printing(every_n_batches=100)
  ]
)
if False:
  pass
  
print("Defined MainLoop");

main_loop.run()




# Alternatively, during test-time
if False:
  labels_out = labels_raw.argmax(axis=1)
  print("labels_out shape", labels_out.shape.tag.test_value)              # array([ 464 ]))

  labels = labels_out.reshape( batch_of_sentences )
  print("labels shape", labels.shape.tag.test_value)                      # array([ 29, 16]))



## Debugging computation overall :
cg = ComputationGraph([cost])

if False:
  #print(cg.variables)
  #print( VariableFilter(roles=[OUTPUT])(cg.variables) )

  #dir(cg.outputs)
  #np.shape(cg.outputs)

  #mlp = MLP([Softmax()], [embedding_dim*2, labels_size],
  #          weights_init=IsotropicGaussian(0.01),
  #          biases_init=Constant(0))
  #mlp.initialize()

  #fork = Fork([name for name in encoder.prototype.apply.sequences if name != 'mask'])
  #fork.input_dim = dimension
  #fork.output_dims = [dimension for name in fork.input_names]
  #print(fork.output_dims)

  cost = aggregation.mean(generator.cost_matrix(x[:, :]).sum(), x.shape[1])
  cost.name = "sequence_log_likelihood"
  model=Model(cost)


#print("TODO :: check on expected input format")
"""
. env/bin/activate
export FUEL_DATA_PATH=~/.fuel
mkdir  ~/.fuel
cd  ~/.fuel
fuel-download mnist
fuel-convert mnist
fuel-download mnist --clear
fuel-info mnist.hdf5

python >>>
from fuel.datasets import MNIST
fuel.config.data_path
mnist = MNIST( u'train' )  # Differs from instructions

s = DataStream.default_stream( mnist, iteration_scheme=ShuffledScheme(mnist.num_examples, 512))
epoch = s.get_epoch_iterator()
e = next(epoch)

>>> e[0].shape
(512, 1, 28, 28)  #So these are the examples, each image being the 1 element of the 2nd index

>>> e[1].shape 
(512, 1)          #So these are the labels

>>> e[2].shape
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: tuple index out of range

"""

#print("TODO :: Find out about numpy.ndarray size changed warning")
"""
Warning :: RuntimeWarning: numpy.ndarray size changed
=> try ::
theano-cache clear 

python >>>
np.version.version
'1.9.2'
np.__file__ 
'/usr/lib64/python2.7/site-packages/numpy/__init__.pyc'

rpm -qa | grep numpy
numpy-1.9.2-1.fc22.x86_64


https://groups.google.com/forum/#!topic/theano-users/A__NVIBYMxA ::

OK, so this is actually due to the new array interface of numpy 1.8 
interacting with Cython. According to [1] and [2], it is harmless. 

One of these posts suggests recompiling the Cython code to get rid of 
the warning, but I'm not sure if it would actually help, and it could 
make the warning appear on the majority of installations with 
numpy < 1.8 instead. 

[1] http://mail.scipy.org/pipermail/numpy-discussion/2012-April/061741.html 
[2] https://mail.python.org/pipermail/cython-devel/2012-January/001848.html 
"""

#print("TODO :: masks for input and output layer")
"""
Cost functions that respect masks for variable-length input (produced with Padding)

https://groups.google.com/forum/#!topic/blocks-users/O-S45G6tpNY
Including target sequence mask in cost function for recurrent network

https://github.com/mila-udem/blocks/issues/653
Cost for recurrent networks

See mask reshape/multiplication for costs somewhere near :
https://github.com/mila-udem/blocks/blob/master/blocks/bricks/sequence_generators.py#L277
"""

#print("TODO :: text reader from CoNLL")
"""
This can be simplified, assuming --docstart-- is a one-word sentence with label 'O'
Should add it to vocab (? maybe <UNK> for one word is good enough)
"""
