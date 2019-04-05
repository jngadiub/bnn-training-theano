import lfc
import lasagne
import keras
import numpy as np
import theano.tensor as T
import quantized_net
from collections import OrderedDict

# load the model from file

# The parameters were already used in the training,
# just need them here for placeholder
args = OrderedDict()
args.activation_bits = 2
args.weight_bits = 1
learning_parameters = OrderedDict()
learning_parameters.activation_bits = args.activation_bits
learning_parameters.weight_bits = args.weight_bits
learning_parameters.alpha = .1
learning_parameters.epsilon = 1e-4
learning_parameters.dropout_in = .2 # 0. means no dropout
learning_parameters.dropout_hidden = .5
learning_parameters.W_LR_scale = "Glorot" # "Glorot" means we are using the          coefficients from Glorot's paper
input = T.matrix('inputs')

mlp = lfc.genLfc(input, 5, learning_parameters)

pfile = np.load('mnist-1w-2a.npz')
params = []
for i in range(len(pfile.keys())):
    params.append(pfile['arr_' + str(i)])

lasagne.layers.set_all_param_values(mlp, params)
