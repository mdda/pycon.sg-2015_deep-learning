import numpy as np

import theano
from theano import tensor

#from blocks import initialization
from blocks.bricks import Identity, Linear, Tanh, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, BaseRecurrent
from blocks.bricks.parallel import Merge
#from blocks.bricks.parallel import Fork

from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import INPUT, WEIGHT, OUTPUT

floatX = theano.config.floatX = 'float32'
# theano.config.assert_no_cpu_op='raise'

theano.config.compute_test_value = 'raise'

vocab_size=4
embedding_dim=40

#hidden_dim=34  # This is a problem... if it is not the same as the embedding
hidden_dim=embedding_dim  # This seems to be the expectation

labels_size=10

max_sentence_length = 29
mini_batch_size = 128
# This becomes the size of the RNN 'output', 
# each place with a (hidden_dim*2) vector (x2 because it's bidirectional)

batch_of_sentences = (mini_batch_size, max_sentence_length)

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

"""
Comments indicate that a reshaping has to be done, so let's think 
about sizes of the arrays...
"""

x = tensor.lmatrix('sentences')

x_mask = tensor.matrix('sentences_mask', dtype=floatX)
#rnn.apply(inputs=input_to_hidden.apply(x), mask=x_mask)

lookup = LookupTable(vocab_size, embedding_dim)
#?? lookup.print_shapes=True

rnn = Bidirectional(
  SimpleRecurrent(dim=hidden_dim, activation=Tanh(),
    weights_init=IsotropicGaussian(0.01),
    biases_init=Constant(0),
  ),
)

### But will need to reshape the rnn outputs to produce suitable input here...
gather = Linear(name='hidden_to_output', 
  input_dim=hidden_dim*2, output_dim=labels_size,
  weights_init=IsotropicGaussian(0.01),
  biases_init=Constant(0)
)

p_labels = Softmax()



## Let's initialize some stuff
lookup.allocate()
print("lookup.params=", lookup.params)

#lookup.weights_init = FUNCTION
#lookup.initialize() 

lookup.params[0].set_value( np.random.normal( scale = 0.1, size=(vocab_size, embedding_dim) ).astype(np.float32) )

rnn.initialize()
gather.initialize()



## Now for the application of these units

# Define the shape of x specifically...  :: the data has format (batch, features).
x.tag.test_value      = np.random.randint(vocab_size, size=batch_of_sentences )
x_mask.tag.test_value = np.random.choice( [0.0, 1.0], size=batch_of_sentences ).astype(np.float32)

#tensor.reshape(x, batch_of_sentences  )
#x = tensor.specify_shape(x_base, batch_of_sentences)
#print("x (new) shape", x.shape.eval())

print("x (new) shape", x.shape.tag.test_value)

embedding = lookup.apply(x)

rnn_outputs = rnn.apply(embedding, mask=x_mask)

print("rnn_outputs shape", np.shape(rnn_outputs).tag.test_value)
#('rnn_outputs shape', array([ 29, 128,  80]))

### So : Need to reshape the rnn outputs to produce suitable input here...
# Convert a tensor here into a long stream of vectors

rnn_outputs_reshaped = rnn_outputs.reshape( (max_sentence_length*mini_batch_size, hidden_dim*2) )

print("rnn_outputs_reshaped shape", np.shape(rnn_outputs_reshaped).tag.test_value)
#('rnn_outputs_reshaped shape', array([3712,   80]))

raw_labels = gather.apply(rnn_outputs_reshaped)  # This is pre-softmaxing

print("raw_labels shape", np.shape(raw_labels).tag.test_value)
#('raw_labels shape', array([ 29*128,  10]))

y_hat = p_labels.apply(raw_labels)  # This is a list of label probabilities

print("y_hat shape", np.shape(y_hat).tag.test_value)
#('y_hat shape', array([3712,   10]))  # -- so this is an in-place rescaling

y = tensor.lmatrix('targets')          # This is a symbolic vector of ints (implies one-hot in categorical_crossentropy)
y.tag.test_value = np.random.randint( vocab_size, size=batch_of_sentences )

print("y shape", y.shape.tag.test_value)
# ('y shape', array([ 29, 128]))

"""
class CategoricalCrossEntropy(Cost):
    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        cost = tensor.nnet.categorical_crossentropy(y_hat, y).mean()
        return cost
"""
#cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

## Version with mask : 
cce = tensor.nnet.categorical_crossentropy(y_hat, y.flatten())
y_mask = x_mask.flatten()
cost = (cce * y_mask) / y_mask.sum()  # elementwise multiple, followed by scaling 


## Less explicit version
#mlp = MLP([Softmax()], [hidden_dim, labels_size],
#          weights_init=IsotropicGaussian(0.01),
#          biases_init=Constant(0))

#print(encoder.prototype.apply.sequences)
#dir(encoder.prototype.apply.sequences)

#probs = mlp.apply(encoder.apply(lookup.apply(x)))
#cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
#cg = ComputationGraph([cost])




# Alternatively, during test-time

labels_list = raw_labels.argmax(axis=1)

print("labels_list shape", np.shape(labels_list).tag.test_value)
#('labels_list shape', array([3712]))

labels = labels_list.reshape( batch_of_sentences )

print("labels shape", np.shape(labels).tag.test_value)
#('labels shape', array([ 29, 128]))




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

#print("TODO :: masks for input layer")
"""
Cost functions that respect masks for variable-length input (produced with Padding)

https://groups.google.com/forum/#!topic/blocks-users/O-S45G6tpNY
Including target sequence mask in cost function for recurrent network

https://github.com/mila-udem/blocks/issues/653
Cost for recurrent networks

See mask reshape/multiplication for costs somewhere near :
https://github.com/mila-udem/blocks/blob/master/blocks/bricks/sequence_generators.py#L277
"""

print("TODO :: text reader from CoNLL")
"""
This can be simplified, assuming --docstart-- is a one-word sentence with label 'O'
Should add it to vocab (? maybe <UNK> for one word is good enough)
"""
