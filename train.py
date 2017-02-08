import sys
import os
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from model.model import LM
from model.runner import datarunner
from model.reader import dataset

#Define the class for the model Options
class Options(object):
  def __init__(self):

    #Data
    self.path = "./data/"
    #self.path = "/vol/work2/labeau/lm/data/news/NceNN/"
    self.n = 3
    self.batch_size = 512
    self.train_set = dataset(file_path=self.path, data_path='train', batch_size=self.batch_size, n=self.n + 1, vocab=None, threshold = 5)
    self.test_set = dataset(file_path=self.path, data_path='test', batch_size=self.batch_size, n=self.n + 1, vocab=self.train_set.vocab)
    self.vocab_size = len(self.train_set.vocab)
    self.train_size = len(self.train_set.data)
    self.test_size = len(self.test_set.data)
    self.n_training_steps = self.train_size // self.batch_size
    self.n_testing_steps = self.test_size // self.batch_size
    self.training_sub = 1000

    #Structural choices
    self.emb_dim = 50
    self.hidden_dim = 100

    #Hyperparameters
    self.learning_rate = 0.001
    self.lr_decay = 0.9
    self.epochs = 100
    self.dropout = 0.5
    self.reg = 0.00005

    #Objective: 'nce', 'target', 'norm', else MLE
    self.obj = 'nce'
    #Noise: for nce or target: 'unigram', 'uniform', 'bigram'
    self.noise = 'bigram'

    #NCE/Target
    if self.obj == 'nce' or self.obj == 'target':
      self.k = 100
      self.distortion = 1.0
      self.unique = True
      self.batchedNoise = False
      if not (self.noise == 'bigram'):
        self.noiseDistrib = self.train_set.getNoiseDistrib(self.noise)

    #Auto-normalization
    if self.obj == 'norm':
      self.alpha = 0.1

    #Others
    self.save_path = "saves/"
    self.display_step = 5

  def decay(self):
    self.learning_rate = self.learning_rate * self.lr_decay

#opts = Options()
with tf.Graph().as_default(), tf.Session(
    config=tf.ConfigProto(
      inter_op_parallelism_threads=16,
      intra_op_parallelism_threads=16)) as session: 

  opts = Options()
  print(opts.vocab_size)

  train_runner = datarunner(opts.n, (opts.noise == 'bigram'), opts.vocab_size)
  train_inputs = train_runner.get_inputs(opts.batch_size)
  print(train_inputs[0].get_shape())
  test_runner = datarunner(opts.n)
  test_inputs = test_runner.get_inputs(opts.batch_size)
  with tf.variable_scope("model"):
    model = LM(opts, session, train_inputs)
  with tf.variable_scope("model", reuse=True):
    model_eval = LM(opts, session, test_inputs, training=False)
  tf.initialize_all_variables().run()

  saver = tf.train.Saver()
  tf.initialize_all_variables().run()
  print('Initialized !')

  tf.train.start_queue_runners(sess=session)
  train_gen = opts.train_set.sampler((opts.noise == 'bigram'))
  test_gen = opts.test_set.sampler()
  train_runner.start_threads(session, train_gen)
  test_runner.start_threads(session, test_gen)

  timeFile = str(datetime.now()).replace(' ','_').replace(':','-').replace('.','_')
  results_file = open('./logs/log' + timeFile,'w')
  opts_attr = vars(opts)
  for attr in opts_attr.items():
    if not (str(attr[0]) == 'noiseDistrib'):
      results_file.write(str(attr[0]) + ' : ' + str(attr[1]) + '\n')
  print ("Epoch 0:")
  results_file.write(str(0) + '\n')
  model_eval.call(results_file)
  for ep in xrange(opts.epochs):
    print ("Epoch %i :" % (ep+1))
    results_file.write(str(ep+1) + '\n')
    model.call(results_file)
    model_eval.call(results_file)
    model._options.decay()
    saver.save(session,
               os.path.join(opts.save_path, "model" + timeFile + ".ckpt"),
               global_step=model.global_step)
  results_file.close()
