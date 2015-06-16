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

vocab_size=3
embedding_dim=3
labels_size=10

lookup = LookupTable(vocab_size, embedding_dim)

encoder = Bidirectional(SimpleRecurrent(dim=embedding_dim, activation=Tanh()))

mlp = MLP([Softmax()], [embedding_dim, labels_size],
          weights_init=IsotropicGaussian(0.01),
          biases_init=Constant(0))

print(encoder.prototype.apply.sequences)
#dir(encoder.prototype.apply.sequences)

#combine = Merge(input_dims=dict(), output_dim=labels_size)
#labelled = Softmax( encoder )


x = tensor.imatrix('features')
y = tensor.imatrix('targets')

probs = encoder.apply(lookup.apply(x))
cg = ComputationGraph([probs])

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
