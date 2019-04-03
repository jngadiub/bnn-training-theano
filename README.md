# Training of Binary Networks

Start from BNN training code from Xilinx [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ/tree/master/bnn/src/training)

The code is written in Theano and is based on this original source https://github.com/MatthieuCourbariaux/BinaryNet from the authors of the following publication [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)

Here it is tested and evaulated on the hls4ml jet-tagging [dataset](https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v)

## Installation and setup:

```
conda create --copy --name bnn-xilinx-py27 python=2.7
source activate bnn-xilinx-py27

conda install -n bnn-xilinx-py27 numpy
conda install -n bnn-xilinx-py27 theano
conda install -n bnn-xilinx-py27 lasagne

conda install -n bnn-xilinx-py27 pylearn2
conda install Theano=0.8
conda install nose-parameterized
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
conda install -c conda-forge numpy

#additional packages
conda install h5py
conda install matplotlib
conda install scikit-learn
conda install pandas
#then install numpy again: conda install -c conda-forge numpy
```

## Run

```
python jet-tagging.py -ab 2 -wb 1
```

