# From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
## ==============================================================================

from __future__ import division

import collections
import os
import sys
import time
import random

import numpy as np
import tensorflow as tf

from nltk.util import ngrams
from tensorflow.python.platform import gfile

"""
TODO: Preprocessing online - pour ne pas garder le corpus en memoire
"""

def _read_words(filename):
  with gfile.GFile(filename, "r") as f:
    return f.read().split()

def _read_lines(filename):
  with gfile.GFile(filename, "r") as f:
    return f.readlines()

def _filter_count(count, threshold):
    return ((k, v) for k, v in count if v >= threshold)

def _build_vocab(filename, threshold= 10):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: -x[1])
  words, counts = list(zip(*_filter_count(count_pairs, threshold)))
  word_to_id = dict(zip(words, range(0, len(words))))
  id_to_word = dict(zip(range(0, len(words)), words))
  return word_to_id, id_to_word, [np.int64(i) for i in counts]

def _file_to_word_ids(filename, word_to_id):
  data = _read_lines(filename)
  return [[word_to_id.get(word,len(word_to_id)-1) for word in line.split()] for line in data]

class dataset():
    def __init__(self,
                 file_path,
                 data_path,
                 batch_size=256,
                 n=4,
                 vocab=None,
                 threshold = 5):
      self.path = os.path.join(file_path,data_path)

      if vocab == None:
        self.vocab, self.inv_vocab, self.counts = _build_vocab(self.path, threshold)
      else:
        self.vocab = vocab

      self.n_grams = [ngrams(sent,n) for sent in _file_to_word_ids(self.path, self.vocab)]
      self.data = np.array([n_gram for sent in self.n_grams for n_gram in sent], dtype = 'int64')

      self.n = n     
      self.cpt = 0
      self.tot = len(self.data)
      self.ids = range(self.tot)
      self.batch_size = batch_size
      random.shuffle(self.ids)

    def getNoiseDistrib(self, noise):
      if noise == "uniform":
        return [1]*len(self.vocab)
      elif noise == "unigram":
        return self.counts
      elif noise == "bigram":
        bigrams = [ngrams(sent, 2) for sent in _file_to_word_ids(self.path, self.vocab)]
        bigramData = [bigram for sent in bigrams for bigram in sent]
        bigramCounts = collections.Counter(bigramData)
        sparseCounts = tf.sparse_reorder(tf.SparseTensor(indices=bigramCounts.keys(), values=bigramCounts.values(), shape=[len(self.vocab), len(self.vocab)]))        
        return sparseCounts                

    def getWordDistrib(self, word):
      dic = self.LM[word]
      vec = np.zeros(self.vocab_size)
      vec[dic.keys()] = dic.values()
      vec += self.alpha
      return vec

    def sampler(self, noise=False):
      if noise:
        self.vocab_size = len(self.vocab)
        self.alpha = 1
        self.LM = collections.defaultdict(lambda: collections.defaultdict(int))
        bigrams = [ngrams(sent, 2) for sent in _file_to_word_ids(self.path, self.vocab)]
        bigramData = [bigram for sent in bigrams for bigram in sent]
        for bigram in bigramData:
          self.LM[bigram[0]][bigram[1]] += 1
      while True:
            if (self.cpt+self.batch_size > self.tot):
                self.cpt=0
                random.shuffle(self.ids)
            xlist = list()
            ylist = list()    
            for i in range(self.batch_size):
                self.cpt+=1
                xlist.append(self.data[self.ids[self.cpt-1],0:self.n-1])
                ylist.append(self.data[self.ids[self.cpt-1],self.n-1])
            if noise:
              yield np.array(xlist, dtype ='int64'),np.array(ylist, dtype='int64'),np.array([self.getWordDistrib(context[-1]) for context in xlist], dtype='int64')
            else:  
              yield np.array(xlist, dtype ='int64'),np.array(ylist, dtype='int64')

