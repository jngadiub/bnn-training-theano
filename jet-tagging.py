#BSD 3-Clause License
#=======
#
#Copyright (c) 2017, Xilinx Inc.
#All rights reserved.
#
#Based Matthieu Courbariaux's MNIST example code
#Copyright (c) 2015-2016, Matthieu Courbariaux
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the copyright holder nor the names of its 
#      contributors may be used to endorse or promote products derived from 
#      this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import sys
import os
import time
from argparse import ArgumentParser
import yaml
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import quantized_net
import lfc

#from pylearn2.datasets.mnist import MNIST
#from pylearn2.utils import serial
#from keras.datasets import mnist

from collections import OrderedDict

def makeRoc(features_val, labels, labels_val, predict_test, outputDir):
    
    print('in makeRoc()')
    if 'j_index' in labels: labels.remove('j_index')
    
    #predict_test = model.predict(features_val)
    
    df = pd.DataFrame()
    
    fpr = {}
    tpr = {}
    auc1 = {}
    
    plt.figure()
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = predict_test[:,i]
        
        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])
        
        auc1[label] = auc(fpr[label], tpr[label])
        
        plt.plot(tpr[label],fpr[label],label='%s tagger, AUC = %.1f%%'%(label.replace('j_',''),auc1[label]*100.))
        print("Label",label,"AUC=",auc1[label]*100.)
    
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig('%s/ROC.png'%(outputDir))

    plt.figure()
    for i, label in enumerate(labels):
     df[label] = labels_val[:,i]
     df[label + '_pred'] = predict_test[:,i]
    
     plt.hist(df[label + '_pred'],label='%s tagger'%(label.replace('j_','')),bins=100,normed=1, histtype='step')
    
    plt.legend()
    #plt.semilogy()
    #plt.xlim(0,1)
    plt.savefig('%s/predictions.png'%(outputDir))
    return predict_test

def get_features(options, yamlConfig):
    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]
    
    print(treeArray.shape)
    print(treeArray.dtype.names)
    
    # List of features to use
    features = yamlConfig['Inputs']
    
    # List of labels to use
    labels = yamlConfig['Labels']
    
    # Convert to dataframe
    features_labels_df = pd.DataFrame(treeArray,columns=list(set(features+labels)))
    features_labels_df = features_labels_df.drop_duplicates()
    
    features_df = features_labels_df[features]
    labels_df = features_labels_df[labels]

    # Convert to numpy array
    features_val = features_df.values
    labels_val = labels_df.values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    #X_train = 2* X_train - 1.
    #X_val = 2* X_val - 1.
    #X_test = 2* X_test - 1.
    
    # Binarise the inputs.
    #X_train = np.where(X_train < 0, -1, 1).astype(theano.config.floatX)
    #X_val = np.where(X_val < 0, -1, 1).astype(theano.config.floatX)
    #X_test = np.where(X_test < 0, -1, 1).astype(theano.config.floatX)
    
    # for hinge loss
    y_train = 2*y_train - 1.
    y_test = 2*y_test - 1.
    y_val = 2*y_val - 1.
    
    return X_train, X_test, X_val, y_train, y_test, y_val, labels

## Config module
def parse_config(config_file) :
    
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

if __name__ == "__main__":
    # Parse some command line options
    parser = ArgumentParser(
        description="Train the LFC network on the MNIST dataset")
    parser.add_argument('-ab', '--activation-bits', type=int, default=1, choices=[1, 2],
        help="Quantized the activations to the specified number of bits, default: %(default)s")
    parser.add_argument('-wb', '--weight-bits', type=int, default=1, choices=[1],
        help="Quantized the weights to the specified number of bits, default: %(default)s")
    parser.add_argument('-i','--input'   ,action='store',dest='inputFile'   ,default='data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
    parser.add_argument('-t','--tree'   ,action='store',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_argument('-c','--config'   ,action='store',dest='config', default='train_config_threelayer.yml', help='configuration file')

    args = parser.parse_args()

    learning_parameters = OrderedDict()

    # Quantization parameters
    learning_parameters.activation_bits = args.activation_bits
    print("activation_bits = "+str(learning_parameters.activation_bits))
    learning_parameters.weight_bits = args.weight_bits
    print("weight_bits = "+str(learning_parameters.weight_bits))

    # BN parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    learning_parameters.alpha = .1
    print("alpha = "+str(learning_parameters.alpha))
    learning_parameters.epsilon = 1e-4
    print("epsilon = "+str(learning_parameters.epsilon))
    
    # Training parameters
    num_epochs = 100
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    learning_parameters.dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(learning_parameters.dropout_in))
    learning_parameters.dropout_hidden = .5
    print("dropout_hidden = "+str(learning_parameters.dropout_hidden))
    
    # W_LR_scale = 1.    
    learning_parameters.W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(learning_parameters.W_LR_scale))
    
    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "mnist-%dw-%da.npz" % (learning_parameters.weight_bits, learning_parameters.activation_bits)
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading MNIST dataset...')
    
    yamlConfig = parse_config(args.config)
    X_train, X_test, X_val, y_train, y_test, y_val, labels  = get_features(args, yamlConfig)

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    #mlp = lfc.genLfc(input, 10, learning_parameters)
    mlp = lfc.genLfc(input, 5, learning_parameters)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    # W updates
    W = lasagne.layers.get_all_params(mlp, quantized=True)
    W_grads = quantized_net.compute_grads(loss,mlp)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = quantized_net.clipping_scaling(updates,mlp)
    
    # other parameters updates
    params = lasagne.layers.get_all_params(mlp, trainable=True, quantized=False)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    my_val_loss,my_train_loss,my_val_err,mlp_best = quantized_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path,
            shuffle_parts)

    output = lasagne.layers.get_output(mlp_best, X_test)
    print(output.shape.eval())
    print((output.eval()).shape)
    
    makeRoc(X_test, labels, y_test, output.eval(), "./")

    plt.figure()
    plt.plot(my_val_loss, label='validation')
    plt.plot(my_train_loss, label='train')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("./loss.png")

    my_val_err = np.array(my_val_err)
    my_val_err = my_val_err/100.
    plt.figure()
    plt.plot(my_val_err, label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig("./acc.png")
