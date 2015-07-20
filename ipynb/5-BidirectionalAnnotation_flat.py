import numpy as np

import theano
from theano import tensor

from blocks.bricks import Identity, Linear, Tanh, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, BaseRecurrent

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
from blocks.serialization import load_parameter_values

floatX = theano.config.floatX = 'float32'
# theano.config.assert_no_cpu_op='raise'

theano.config.compute_test_value = 'raise'

#theano.config.optimizer='None'  # Not a Python None
theano.config.optimizer='fast_compile'

run_test = True and False

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

print("Embedding shape :", embedding.shape)                             # (4347, 100)
print("Embedding dtype :", embedding.dtype)                             # float32

## These are set from the contents of the embedding file itself
vocab_size, embedding_dim = embedding.shape

extra_size = 1 # (caps,)

hidden_dim = embedding_dim+extra_size  # RNN units align over total embedding size

labels_size=5 # ( 0 .. 4 inclusive ), see range of CoNLLTextFile.ner values

max_sentence_length = 128 # There's a long sentence in testb... (124 words)
mini_batch_size = 8
num_batches=10000  # For the whole main_loop

# This also defines the maximum size of the RNN 'output' stage - 
# each element being a (hidden_dim*2) vector (x2 because it's bidirectional)
batch_of_sentences = (max_sentence_length, mini_batch_size)  # Since the data_stream has a _transpose

#save_state_path='saved_state.npz'
save_state_path='train.10epochs.cpu'

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
  label2code={
    'O'     :(0, 0), 
    'I-PER' :(0, 1), 
    'I-LOC' :(0, 2), 
    'I-ORG' :(0, 3), 
    'I-MISC':(0, 4), 
    'B-PER' :(1, 1),  # These 'Beginning' labels are not used ('B-PER' does not appear in training or testa)
    'B-LOC' :(1, 2),  # These 'Beginning' labels are not used (others are very infrequent ~ 0.2%)
    'B-ORG' :(1, 3), 
    'B-MISC':(1, 4), 
  }
  _digits = re.compile('\d')
  unknown = None

  def __init__(self, files, dictionary, unknown_token, **kwargs):
    self.files = files
    self.dictionary = dictionary 
    self.unknown = dictionary[unknown_token] # Error if not there
    
    self.code2label = { c[1]:l for (l,c) in self.label2code.items() if c[0]==0 }
    #print(self.code2label)
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
        labels.append(self.label2code[ l[-1] ][1] ) # Use just the second entry as the output target label...
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

if run_test:
  data_paths = ['/home/andrewsm/SEER/external/CoNLL2003/ner/eng.testb',]  # 748Kb file
else:
  data_paths = ['/home/andrewsm/SEER/external/CoNLL2003/ner/eng.train',]  # 3.3Mb file
  
## Achieved result: 50-epochs (GPU) training on eng.train => testb overall scores :
## accuracy:  96.42%; precision:  76.95%; recall:  80.26%; FB1:  78.57

dataset = CoNLLTextFile(data_paths, dictionary=word2code, unknown_token='<UNK>')

data_stream = DataStream(dataset)
data_stream = Filter(data_stream, _filter_long)
#data_stream = Mapping(data_stream, reverse_words, add_sources=("targets",))

data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(mini_batch_size))

#data_stream = Padding(data_stream, mask_sources=('tokens'))            # Adds a mask fields to this stream field, type='floatX'
data_stream = Padding(data_stream, )              # Adds a mask fields to all of this stream's fields, type='floatX'
data_stream = Mapping(data_stream, _transpose)    # Flips stream so that sentences run down columns, batches along rows (strangely)

if False: # print sample for debugging Dataset / DataStream component
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
Comments in google-groups:blocks indicate that a reshaping has to be done, 
so let's think about sizes of the arrays...
"""

x = tensor.matrix('tokens', dtype="int32")

x_mask = tensor.matrix('tokens_mask', dtype=floatX)
#rnn.apply(inputs=input_to_hidden.apply(x), mask=x_mask)

lookup = LookupTable(vocab_size, embedding_dim)

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

# The shape actually depends on the specific batch... (for instance, the last one in an epoch may be smaller)
#rnn_outputs_reshaped = rnn_outputs.reshape( (max_sentence_length*mini_batch_size, hidden_dim*2) )  # not parameterized properly
rnn_outputs_reshaped = rnn_outputs.reshape( (x.shape[0]*x.shape[1], hidden_dim*2) )
print("rnn_outputs_reshaped shape", rnn_outputs_reshaped.shape.tag.test_value)   #array([464, 202]))

labels_raw = gather.apply(rnn_outputs_reshaped)  # This is pre-softmaxing
print("labels_raw shape", labels_raw.shape.tag.test_value)              # array([ 464, 5]))


def examine_embedding(embedding):
  e = embedding.copy()
  print("Examine Embedding Shape : ", e.shape)
  
  norms = np.apply_along_axis(np.linalg.norm, 1, e).reshape( (e.shape[0],1) )  # normalize all vectors in the embedding
  #print("Examine Embedding norms Shape : ", norms.shape)
  e = e / norms
  
  for token_target in ["give", "wait", "book", "turkey", "angeles", "he", "further", "do", "monday", "NUMBER", ]:
    token_i = word2code.get(token_target, None)
    if token_i is None:
      print("Found token '%s' NOT FOUND " % (token_target,))
      continue
    #print("Found token '%s' at %d" % (token_target, token_i))
    
    token_v = e[token_i]
    #print("self-cosine similarity %f" % (np.dot(token_v,token_v)))
    
    all_similarities = np.dot(e, token_v)
    #print("overall similarity shape: ", all_similarities.shape)  # a 1-d array
    
    sorted_similarities = sorted( enumerate(all_similarities), key=lambda (i,v): -v)
    print("Top Similarities for %10s @ %4d:" % (token_target,token_i, ), 
      map(lambda (i,v): "%s %.1f%%" % (code2word[i],v*100.), 
        sorted_similarities[1:4] # Element [0] is token itself
      )  
    )
  
  #exit(0)

if not run_test:  # i.e. do training phase
  label_probs = p_labels.apply(labels_raw)               # This is a list of label probabilities
  print("label_probs shape", label_probs.shape.tag.test_value)          # array([ 464, 5]))
  # -- so :: this is an in-place rescaling

  y = tensor.matrix('labels', dtype="int32")   # This is a symbolic vector of ints (implies one-hot in categorical_crossentropy)
  y.tag.test_value = np.random.randint( labels_size, size=batch_of_sentences).astype(np.int32)

  print("y shape", y.shape.tag.test_value)                              # array([ 29, 16]))
  print("y.flatten() shape", y.flatten().shape.tag.test_value)          # array([464]))
  print("y.flatten() dtype", y.flatten().dtype)                         # int32

  examine_embedding(lookup.W.get_value())

  """
  class CategoricalCrossEntropy(Cost):
      @application(outputs=["cost"])
      def apply(self, y, y_hat):
          cost = tensor.nnet.categorical_crossentropy(y_hat, y).mean()
          return cost
  """
  #cost = CategoricalCrossEntropy().apply(y.flatten(), label_probs)

  ## Version with mask : 
  #cce = tensor.nnet.categorical_crossentropy(label_probs, y.flatten()) # We can go deeper...
  cce = tensor.nnet.crossentropy_categorical_1hot(label_probs, y.flatten())

  y_mask = x_mask.flatten()
  print("y_mask shape", y_mask.shape.tag.test_value)                      # array([464]))
  print("y_mask dtype", y_mask.dtype)                                     # float32

  cost = (cce * y_mask).sum() / y_mask.sum()             # elementwise multiple, followed by scaling 
  cost.name='crossentropy_categorical_1hot_masked'

  print("Created explicit cost:");
  print(cost)

  # Define the training algorithm.
  cg = ComputationGraph(cost)

  #print("Created ComputationGraph, variables:");
  #print(cg.variables)

  print("Created ComputationGraph, parameters:");
  #print(cg.parameters)
  for p in cg.parameters:
    print(str(p), p.dtype, p.shape.tag.test_value)

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
      #FinishAfter(after_n_epochs=50),
      FinishAfter(after_n_epochs=50),
      
      # Saving the model and the log separately is convenient,
      # because loading the whole pickle takes quite some time.
      #Checkpoint(save_state_path, every_n_batches=2000*10, save_separately=["model", "log"]),
      Checkpoint(save_state_path, every_n_epochs=10, save_separately=["model", "log"]),
      Printing(every_n_batches=500)
    ]
  )
  print("Defined MainLoop")
  
  main_loop.run()

else:
  # Alternatively, during test-time
  labels_out = labels_raw.argmax(axis=1)
  print("labels_out shape", labels_out.shape.tag.test_value)            # array([ 464 ]))

  labels = labels_out.reshape( (x.shape[0], x.shape[1]) )               # Again, this depends on the batch
  print("labels shape", labels.shape.tag.test_value)                    # array([ 29, 16]))

  if False:
    # Define the testing algorithm.
    cg = ComputationGraph(labels)

    #print("Created ComputationGraph, variables:");
    #print(cg.variables)

    print("Created ComputationGraph, parameters:");
    #print(cg.parameters)
    for p in cg.parameters:
      print(str(p), p.shape, p.dtype)

    print("Created ComputationGraph, inputs:");
    print(cg.inputs)
  
  # Strangely, all the examples use : DataStreamMonitoring in MainLoop

  model = Model(labels)
  print("Model.dict_of_inputs():");
  print(model.dict_of_inputs())
  print("Model list inputs:");
  print([ v.name for v in model.inputs])

  ## Model loading from saved file
  model.set_parameter_values(load_parameter_values(save_state_path))  

  examine_embedding(lookup.W.get_value())
    
  label_ner = model.get_theano_function()
  print(model.inputs)
  print("printed label_ner.params")

  for test_data in data_stream.get_epoch_iterator():
    ordered_batch = test_data[0:3]   # Explicitly strip off the pre-defined labels
    #print(ordered_batch)
    
    results = label_ner(*ordered_batch)
    #print(results)  # This is a pure array of labels
    
    inputs = _transpose(ordered_batch)
    for tokens, mask, labels in zip(inputs[0], inputs[1], np.transpose(results)):
      #print(labels)
      for (token, m, label) in zip(tokens, mask, labels.tolist()):
        if m<0.1: 
          break  # once the mask is off, no need to keep going on this sentence
        #print(token, label[0]) 
        print("%s %s" % (code2word[token], dataset.code2label[ label[0] ]) ) 
      # End of sentence
      print("")
    # End of batch
  print("")
  # End of input

if False:
  ## Debugging computation overall :
  cg = ComputationGraph([cost])
  
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

