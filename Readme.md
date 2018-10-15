Tensorflow Coding Examples
===
# Author
GA WU, PhD candidate, University of Toronto.

# Tensorflow Version
This code was written in Tensorflow beta version 0.12c. If you need to run it in Tensorflow 1.3 or later version. Please see the [PURE_PYTHON_CODE](PURE_PYTHON_CODE)


# Introduction
Tensorflow is not only an well designed deep learning toolbox, but also a standard symbolic programming framework. In this repository, we show how to use tensorflow to do classical planning task on deterministic, continous action, continous space problems. Our investigation has two phrases.
1. We first parse domains that descripted by standard domain description language into Tensorflow, and directly do planning on the hard coded domains. The domain description language can be: PDDL, PDDL2 and RDDL. We think RDDL has much powerful expressive ability, so in the code, if there is no specific description, we use RDDL.
2. Because of the ability of learning arbitrary function, neural network can learn the transition function and reward function directly from observed data. We then investigate the ability to learn the model and do planning on learned, approximated model to solve real problem

# What is in the repository
This repository contains following implementations

1. Hard coded domains: There are three hard coded domains in the [HARD_CODED_DOMAINS](HARD_CODED_DOMAINS) folder.
2. Deep learned model: We provide a framework [DEEP_LEARNED_PLANNING](DEEP_LEARNED_PLANNING) based on Tensorflow rnn cell that allows to learn and plan through sampled data. (This code is re-wroten in another repository, see below)
3. Visualization functions:[VIZ](VIZ) Show the behavious of result from planner.

# Planning With Neural Network Trained Transition Function
Because the tensorflow upgrade recent days, the old method to customize RNN cell has been deprecated. We re-wrote the tensorflow planner for trained transition function in the following repository. Please please directly copy the command in repository to test its functionality.

[TensorFlow Planner for Trained Transition Function](https://github.com/wuga214/PlanningThroughTensorFlow)
