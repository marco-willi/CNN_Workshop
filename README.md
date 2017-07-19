# Convolutional Neural Networks (CNNs)- Workshop

## Overview
This workshop is intended to be run on a machine with GPU support & uses jupyter notebooks to interactively run the code. However, also a non-GPU (CPU) version is available which runs fairly quick on a standard pc / laptop.

The goal of the workshop is to use Keras to build a CNN on a small data set. The idea is to become familiar with how to set up Keras and to train a model.

## AWS GPU instance configuration
If you want to set up an AWS GPU instance, the neccessary steps are described in:
* Part1_install_aws.sh
* Part2_install_MANUAL_aws.sh
* Part3_install_aws.sh

If you just want to run it on your local machine (CPU version) you can use this docker image:
* docker run -it tensorflow/tensorflow:nightly-devel-gpu-py3 bash
* pip install keras

## Pre requisites
To run model with Cats & Dogs you may get the images here here:
https://www.microsoft.com/en-us/download/details.aspx?id=54765

To run model with MNIST data (CPU version), the data can be downloaded from within Keras.

To run model with Elephant / Zebra data, a specific Zooniverse AMI has to be used (not available for the public).

# Workshop files
The GPU notebook is: _CNN_Workshop_GPU.ipynb_

The CPU notebook is: _CNN_Workshop_CPU.ipynb_

## Resources / Links
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb
