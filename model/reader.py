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

import pickle
from itertools import islice
from functools import partial
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

def _filter_vocab(vocab, threshold):
    if threshold == 0:
        return vocab
    else:
        return dict((k, v) for k, v in vocab.iteritems() if v < threshold)

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
          yield np.array(xlist, dtype ='int64'),np.array(ylist, dtype='int64'), np.array([self.getWordDistrib(context[-1]) for context in xlist], dtype='int64')
        else:  
          yield np.array(xlist, dtype ='int64'),np.array(ylist, dtype='int64')

class datasetQ():
  def __init__(self,
               dir_path,
               data_file,
               vocab_file,
               bigram = False,
               context_length = 3,               
               word_vocab_threshold = 0,
               bigram_threshold = 0):
    """
    Caution: this iterator will open the file and process lines on the fly before
    yielding them, (to be able to work with files too big to fit in memory
    which implies there is no data shuffling. Training data must be shuffled using:
    shuf --head-count=NB_SEQ train
    """
    self.path = os.path.join(dir_path, data_file)
    self.len_data_path = os.path.join(dir_path, data_file + str(context_length) + '.len.pkl')
    self.vocab_data_path = os.path.join(dir_path, vocab_file + str(context_length) +  '.vocab.pkl')
    
    self.n = context_length + 1
    # If the vocabulary is not already saved:
    if not os.path.exists(self.vocab_data_path):
      self._file_to_vocab()

    if not os.path.exists(self.len_data_path):
      self._file_to_length()

    with open(self.len_data_path, 'r') as _file:
      [tot] = pickle.load(_file)
    with open(self.vocab_data_path, 'r') as _file:
      [word_to_id, word_counts] = pickle.load(_file)
    self.word_to_id = _filter_vocab(word_to_id, word_vocab_threshold)   
    if word_vocab_threshold == 0:
      word_vocab_threshold = len(self.word_to_id)
    if (bigram_threshold >= word_vocab_threshold) or (bigram_threshold == 0):
      self.bigramThreshold = word_vocab_threshold
    else:
      self.bigramThreshold = bigram_threshold
    self.uniform = [1]*word_vocab_threshold
    self.unigram = list(word_counts[:word_vocab_threshold])
    self.tot = tot
    
    if bigram:
      self.bigram_data_path = os.path.join(dir_path, vocab_file + str(self.bigramThreshold) + '.bigram.pkl')
      if not os.path.exists(self.bigram_data_path):
        self._file_to_bigram()

      with open(self.bigram_data_path, 'r') as _file:
        [self.LM] = pickle.load(_file)

  def _file_to_length(self):
    datalen = 0
    with open(self.path) as train_file:
      for line in train_file:
        seq = line.strip().split()
        #datalen += len(ngrams(seq, self.n))
        datalen += sum(1 for _ in ngrams(seq, self.n))
    with open(self.len_data_path, 'w') as len_file:
      pickle.dump([datalen], len_file) 

  def _file_to_vocab(self):
    # Get words, find longuest sentence and longuest word
    data = []
    with open(self.path) as train_file:
      for line in train_file:
        seq = line.strip().split()
        data.append(seq)
    # Get complete word vocabulary
    word_counter = collections.Counter([word for seq in data for word in seq])
    word_pairs = sorted(word_counter.items(), key=lambda x: -x[1])
    words, w_counts = list(zip(*word_pairs))
    word_to_id = dict(zip(words, range(0, len(words)+0)))
    with open(self.vocab_data_path, 'w') as vocab_file:
      pickle.dump([word_to_id, w_counts], vocab_file)

  def _file_to_bigram(self):
    data = []
    bigram_voc = _filter_vocab(self.word_to_id, self.bigramThreshold)
    with open(self.path) as train_file:
      for line in train_file:
        seq = [bigram_voc.get(word, len(bigram_voc)-1) for word in line.strip().split()]
        data.extend(ngrams(seq, 2))
    LM = collections.defaultdict(partial(collections.defaultdict, int))
    for bigram in data:
      LM[bigram[0]][bigram[1]] += 1
    with open(self.bigram_data_path, 'w') as bigram_file:
      pickle.dump([LM], bigram_file)

  def getWordDistrib(self, word, alpha = 1):
    if word < self.bigramThreshold:
      dic = self.LM[word]
      vec = np.zeros(self.bigramThreshold)
      vec[dic.keys()] = dic.values()
      vec += alpha
    else:
      vec = np.array(self.uniform[:self.bigramThreshold])
    return vec

  """
  def getBatchDistrib(self, batch, alpha = 1):
    finalVec = np.zeros(self.bigramThreshold)
    for word in batch:
      if word < self.bigramThreshold:
        dic = self.LM[word]
        vec = np.zeros(self.bigramThreshold)
        vec[dic.keys()] = dic.values()
        vec += alpha
        finalVec += vec
      else:
        vec = np.array(self.uniform[:self.bigramThreshold])
        finalVec += vec
      self.unigram = finalVec
  """
  
  def sampler(self, batch_size, bigram=False, bigram_sum=False):
    with open(self.path) as _file:
      while True:
        to_be_read = list(islice(_file, batch_size))
        if len(to_be_read) < batch_size:
          _file.seek(0)
          to_be_read = list(islice(_file, batch_size))
        """
        TODO: one loop for word/char faster than two loops with list comprehensions ?
        """
        n_grams = [ngrams(sent.strip().split(), self.n) for sent in to_be_read]
        word_train_tensor = np.array([[self.word_to_id.get(w, len(self.word_to_id)-1) for w in n_gram] for sent in n_grams for n_gram in sent], dtype = 'int32')
        x = word_train_tensor[:,:-1]
        y = word_train_tensor[:,-1]
        if bigram:
          yield x, y, np.array([self.getWordDistrib(context[-1]) for context in x], dtype='int64')
        else:
          yield x, y 
