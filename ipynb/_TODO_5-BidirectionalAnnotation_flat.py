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

theano.config.floatX = 'float32'
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

batch_of_sentences = (max_sentence_length, mini_batch_size)

"""
Cost functions that respect masks for variable-length input (produced with Padding)

https://groups.google.com/forum/#!topic/blocks-users/O-S45G6tpNY
Including target sequence mask in cost function for recurrent network

https://github.com/mila-udem/blocks/issues/653
Cost for recurrent networks

See mask reshape/multiplication for costs somewhere near :
https://github.com/mila-udem/blocks/blob/master/blocks/bricks/sequence_generators.py#L277
"""

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

x = tensor.lmatrix('data')

lookup = LookupTable(vocab_size, embedding_dim)
#?? lookup.print_shapes=True

rnn = Bidirectional(SimpleRecurrent(dim=hidden_dim, activation=Tanh()))

### But will need to reshape the rnn outputs to produce suitable input here...
gather = Linear(name='hidden_to_output', input_dim=hidden_dim*2, output_dim=labels_size)

### But will need to reshape the rnn outputs to produce suitable input here...
labels = Softmax()


## Let's initialize some stuff
lookup.allocate()
print("lookup.params=", lookup.params)

#lookup.weights_init = FUNCTION
#lookup.initialize() 
#lookup.params[0].set_value( np.random.normal( scale = 0.1, size=(vocab_size, embedding_dim) ) )
lookup.params[0].set_value( np.random.normal( scale = 0.1, size=(vocab_size, embedding_dim) ).astype(np.float32) )

## Now for the application of these units

# Define the shape of x specifically... 
# looks like it should be max_sentence_length rows and mini_batch_size columns

x.tag.test_value = np.random.randint(vocab_size, size=batch_of_sentences )

#tensor.reshape(x, batch_of_sentences  )
#x = tensor.specify_shape(x_base, batch_of_sentences)
#print("x (new) shape", x.shape.eval())

print("x (new) shape", x.shape.tag.test_value)
print("x (new) shape", np.shape(x).tag.test_value)

embedding = lookup.apply(x)

rnn_outputs = rnn.apply(embedding)

print("rnn_outputs shape", np.shape(rnn_outputs).tag.test_value)
#('rnn_outputs shape', array([ 29, 128,  80]))


### But will need to reshape the rnn outputs to produce suitable input here...
rnn_outputs_reshaped = rnn_outputs.reshape( (max_sentence_length*mini_batch_size, hidden_dim*2) )
#rnn_outputs_reshaped = rnn_outputs
print("rnn_outputs_reshaped shape", np.shape(rnn_outputs_reshaped).tag.test_value)

pre_softmax = gather.apply(rnn_outputs_reshaped)

print("pre_softmax shape", np.shape(pre_softmax).tag.test_value)
#('pre_softmax shape', array([ 29, 128,  10]))

# Received a tensor here...
y_hat = labels.apply(pre_softmax)

#print("y_hat shape", np.shape(y_hat).tag.test_value)
print("y_hat shape", np.shape(y_hat).tag.test_value)


y = tensor.lmatrix('targets')
y.tag.test_value = np.random.randint( vocab_size, size=batch_of_sentences )
#tensor.reshape(y, batch_of_sentences )
print("y shape", y.shape.tag.test_value)

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

## Less explicit version
#mlp = MLP([Softmax()], [hidden_dim, labels_size],
#          weights_init=IsotropicGaussian(0.01),
#          biases_init=Constant(0))





#print(encoder.prototype.apply.sequences)
#dir(encoder.prototype.apply.sequences)

#combine = Merge(input_dims=dict(), output_dim=labels_size)
#labelled = Softmax( encoder )


#probs = encoder.apply(lookup.apply(x))
#cg = ComputationGraph([probs])

#probs = mlp.apply(encoder.apply(lookup.apply(x)))
#cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
#cg = ComputationGraph([cost])

#print(cg.variables)
print( VariableFilter(roles=[OUTPUT])(cg.variables) )

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

