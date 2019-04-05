import lasagne
import keras
import numpy as np
import quantized_net
import sys
import os
sys.path.append('../keras-training-clean/layers')
from quantized_ops import binary_tanh
from quantized_layers import BinaryDense

def lasagneToKeras(lasagneModel):
    model = keras.models.Sequential()
    for layer in lasagne.layers.get_all_layers(lasagneModel):
        addLayer(model, layer)
    return model

def addLayer(model, lasagneLayer):
    # Call the method to convert this type of layer
    fconverter = {lasagne.layers.input.InputLayer : convertInputLayerToKeras,
                  quantized_net.DenseLayer : convertDenseLayerToKeras,
                  lasagne.layers.normalization.BatchNormLayer : convertBatchNormLayerToKeras,
                  lasagne.layers.special.NonlinearityLayer : convertActivationLayerToKeras}
    assert type(lasagneLayer) in fconverter.keys(), str(type(lasagneLayer)) + " is not supported."
    fconverter[type(lasagneLayer)](model, lasagneLayer)

def convertInputLayerToKeras(model, lasagneLayer):
    model.add(keras.layers.InputLayer(input_shape=(lasagneLayer.output_shape[1], )))

def convertDenseLayerToKeras(model, lasagneLayer):
    ''' Convert a quantized_net DenseLayer to a Keras layer '''
    # Initialise a layer with the right shape
    #model.add(keras.layers.Dense(lasagneLayer.output_shape[1], input_shape=(lasagneLayer.input_shape[1], )))
    model.add(BinaryDense(lasagneLayer.output_shape[1], input_shape=(lasagneLayer.input_shape[1], ), use_bias=False))
    # Add the weights from the trained lasagneLayer
    # The have the same order in lasagne and keras: [weight, bias]
    #model.layers[-1].set_weights(map(lambda x : x.get_value(), lasagneLayer.get_params()))
    # No bias for binary
    model.layers[-1].set_weights([lasagneLayer.W.get_value()])

def convertBatchNormLayerToKeras(model, lasagneLayer):
    # According to http://faroit.com/keras-docs/1.2.2/layers/normalization/
    # the order of parameters is [gamma, beta, mean, std]
    ll = lasagneLayer
    model.add(keras.layers.BatchNormalization(epsilon=ll.epsilon))
    weights = map(lambda x : x.get_value(), [ll.gamma, ll.beta, ll.mean, ll.inv_std])
    # lasagne stores 1 / sqrt(sigma^2 + epsilon), keras needs sigma
    # so transform
    weights[-1] = 1 / weights[-1] ** 2 - ll.epsilon
    model.layers[-1].set_weights(weights)

def convertActivationLayerToKeras(model, lasagneLayer):
    actconvert = {quantized_net.FixedHardTanH : binary_tanh}
    assert type(lasagneLayer.nonlinearity) in actconvert.keys(), str(type(lasagnedLayer.nonlinearity)) + " is not supported"
    model.add(keras.layers.Activation(actconvert[type(lasagneLayer.nonlinearity)]))

