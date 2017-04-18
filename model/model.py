import sys
import os
import math
import time

import numpy as np
import tensorflow as tf
import tfutils

from tensorflow.python.ops.nn_impl import _compute_sampled_logits

#Define the class for the Language Model
class LM(object):

  def __init__(self, options, session, inputs, training=True):
    self._options = options
    self._session = session
    self._training = training
    self._noiseDistrib = self._options.noiseDistrib
    if (self._options.noise == 'bigram' and self._training):
      self._examples, self._labels, self._currentNoiseDistrib = inputs
      self._currentNoiseDistrib = tf.concat(axis=1, values=[self._currentNoiseDistrib, tf.ones([self._options.batch_size, self._options.vocab_size - self._options.train_set.bigramThreshold], dtype='int64')])
      self._meanNoiseDistrib = tf.reduce_sum(self._currentNoiseDistrib, 0)
    else: 
      self._examples, self._labels = inputs
    self.build_graph()

  def forward(self, examples, labels):
    self.global_step = tf.Variable(0, name="global_step")

    #Embedding Layer
    self._emb = tf.get_variable(
      name='emb',
      shape=[self._options.vocab_size, self._options.emb_dim],
      initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.vocab_size)) ))

    embeddings = tf.nn.embedding_lookup(self._emb,tf.reshape(examples, [-1]))
    input_emb = tf.reshape(embeddings,[self._options.batch_size, self._options.emb_dim * self._options.n])
    # Batch normalization
    if self._options.batch_norm:
      self.batch_normalizer = tfutils.batch_norm()
      input_emb = self.batch_normalizer(input_emb, self._training)

    #Hidden Layer
    self._h_weights = tf.get_variable(
      name='h_weights',
      shape=[self._options.emb_dim * self._options.n, self._options.hidden_dim],
      initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.emb_dim * self._options.n)) ))
    weight_decay = tf.nn.l2_loss(self._h_weights)
    tf.add_to_collection('losses', weight_decay)

    self._h_biases = tf.get_variable(
      name='h_biases',
      shape=[self._options.hidden_dim],
      initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.emb_dim * self._options.n)) ))
    weight_decay = tf.nn.l2_loss(self._h_biases)
    tf.add_to_collection('losses', weight_decay)

    if self._training and self._options.dropout < 1.0:
      hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(input_emb, self._h_weights) + self._h_biases), self._options.dropout)
    else:
      hidden = tf.nn.relu(tf.matmul(input_emb, self._h_weights) + self._h_biases)

    self._output_weights = tf.get_variable(
      name="output_weights",
      shape= [self._options.hidden_dim, self._options.vocab_size],
      initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)) ))
    self._output_biases = tf.get_variable(
      name="output_biases",
      shape=[self._options.vocab_size],
      initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)) ))

    return hidden 

  def blackOut_loss(self, hidden, labels):
    labels_ext = tf.expand_dims(labels, 1)
    #Same noise samples for the whole batch: we can compute nce_loss by batch with the usual function
    (negative_samples,
     true_expected_counts,
     sampled_expected_counts) = tf.nn.fixed_unigram_candidate_sampler(labels_ext,
                                                                      1,
                                                                      self._options.k,
                                                                      self._options.unique,
                                                                      self._options.vocab_size,
                                                                      distortion=self._options.distortion,
                                                                      num_reserved_ids=0,
                                                                      unigrams=self._noiseDistrib,
                                                                      name='nce_sampling')
    logits, labels_nce = _compute_sampled_logits(tf.transpose(self._output_weights),
                                                 self._output_biases,
                                                 labels_ext,
                                                 hidden,
                                                 self._options.k,
                                                 self._options.vocab_size,
                                                 num_true=1,
                                                 sampled_values= (negative_samples,
                                                                  true_expected_counts,
                                                                  sampled_expected_counts),
                                                 subtract_log_q= True,
                                                 remove_accidental_hits = True,
                                                 name='nce_loss_1')
    #Compute the 'noise offset' for each example - adapted logSumExp for numerical stability 
    sampled_logits = tf.slice(logits,
                              [0, 1],
                              [self._options.batch_size, self._options.k])
    maxes = tf.reduce_max(sampled_logits, 1, keep_dims=True)
    sampled_logits_without_maxes = sampled_logits - maxes
    noise_offset = tf.expand_dims(tf.squeeze(maxes, [-1]) + tf.log(tf.reduce_sum(tf.exp(sampled_logits_without_maxes), 1)), 1)
    
    logits -= noise_offset

    sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_nce, name="nce_loss_2")
    positive_score = sampled_losses[:,0]
    true_loss = tf.reduce_mean(positive_score, name='true_mean')
    nce_score = tf.reduce_sum(sampled_losses, 1)
    loss = tf.reduce_mean(nce_score, name='nce_mean')
    noise_ratio = tf.reduce_mean(noise_offset)
    if self._options.freqScores:
      losses = tfutils._compute_ranged_scores(nce_score, labels, self._options.ranges)
      loss = [loss] + losses
      positive_loss =  tfutils._compute_ranged_scores(positive_score, labels, self._options.ranges)
      true_loss = [true_loss] + positive_loss
      noise_ratios = tfutils._compute_ranged_scores(tf.squeeze(noise_offset), labels, self._options.ranges)
      noise_ratio = [noise_ratio] + noise_ratios
    return loss, true_loss, noise_ratio

  def nce_loss(self, hidden, labels):
    labels_ext = tf.expand_dims(labels, 1)
    #Same noise samples for the whole batch: we can compute nce_loss by batch with the usual function
    if self._options.batchedNoise:
      (negative_samples,
       true_expected_counts,
       sampled_expected_counts) = tf.nn.fixed_unigram_candidate_sampler(labels_ext,
                                                                        1,
                                                                        self._options.k,
                                                                        self._options.unique,
                                                                        self._options.vocab_size,
                                                                        distortion=self._options.distortion,
                                                                        num_reserved_ids=0,
                                                                        unigrams=self._noiseDistrib,
                                                                        name='nce_sampling')
      logits, labels_nce = _compute_sampled_logits(tf.transpose(self._output_weights),
                                                   self._output_biases,
                                                   labels_ext,
                                                   hidden,
                                                   self._options.k,
                                                   self._options.vocab_size,
                                                   num_true=1,
                                                   sampled_values= (negative_samples,
                                                                    true_expected_counts,
                                                                    sampled_expected_counts),
                                                   subtract_log_q= True,
                                                   remove_accidental_hits = True,
                                                   name='nce_loss_1')

      # If we use a mean bigram distribution over the batch, reweight logits according only to their own context
      if self._options.noise == "bigram":
        sampled_logits = tf.slice(logits,
                                  [0, 1],
                                  [self._options.batch_size, self._options.k])
        
        proba = tf.nn.log_softmax(tf.log(self._currentNoiseDistrib), dim = 0)
        idx = tf.cast(tf.range(self._options.batch_size), dtype='int64')
        idx = tf.reshape(idx, [-1, 1])
        idx = tf.tile(idx, [1, self._options.k])
        idx = tf.reshape(idx, [-1])
        sampled_indexes = tf.transpose(tf.pack([idx, negative_samples]))
        logits -= tf.gather_nd(proba, sampled_indexes)
        
      sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_nce, name="nce_loss_2")
      positive_score = sampled_losses[:,0]
      true_loss = tf.reduce_mean(positive_score, name='true_mean')
      nce_score = tf.reduce_sum(sampled_losses, 1)
      loss = tf.reduce_mean(nce_score, name='nce_mean')
    #Different noise samples for each batch
    else:
      # Sample k * batch_size negative examples         
      (negative_samples,
       true_expected_counts,
       sampled_expected_counts) = tf.nn.fixed_unigram_candidate_sampler(labels_ext,
                                                                        1,
                                                                        self._options.k * self._options.batch_size,
                                                                        False,
                                                                        #self._options.unique,
                                                                        self._options.vocab_size,
                                                                        distortion=self._options.distortion,
                                                                        num_reserved_ids=0,
                                                                        unigrams=self._noiseDistrib,
                                                                        name='nce_sampling')
      sampled_expected_counts = tf.scalar_mul(1.0 / self._options.batch_size, sampled_expected_counts)
      true_expected_counts = tf.scalar_mul(1.0 / self._options.batch_size, true_expected_counts)
      
      # And processing that goes with     
      logits, labels_nce = tfutils._compute_sampled_logits_by_batch(tf.transpose(self._output_weights),
                                                                    self._output_biases,
                                                                    hidden,
                                                                    labels_ext,
                                                                    self._options.k * self._options.batch_size,
                                                                    self._options.k,
                                                                    self._options.vocab_size,
                                                                    num_true=1,
                                                                    sampled_values= (negative_samples,
                                                                                     true_expected_counts,
                                                                                     sampled_expected_counts),
                                                                    subtract_log_q= True,
                                                                    name='nce_loss_1_batched')      
      sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_nce, name="nce_loss_2_batched")
      positive_score = sampled_losses[:,0]
      true_loss = tf.reduce_mean(positive_score, name='true_mean')
      nce_score = tf.reduce_sum(sampled_losses, 1)
      loss = tf.reduce_mean(nce_score, name='nce_mean')

    if self._options.freqScores:
      losses = tfutils._compute_ranged_scores(nce_score, labels, self._options.ranges)
      loss = [loss] + losses
      positive_loss =  tfutils._compute_ranged_scores(positive_score, labels, self._options.ranges)
      true_loss = [true_loss] + positive_loss
    return loss, true_loss

  def target_loss(self, hidden, labels):
    ext_labels = tf.expand_dims(labels, 1)    
    (negative_samples,
     true_expected_counts,
     sampled_expected_counts) = tf.nn.learned_unigram_candidate_sampler(ext_labels,
                                                                        1,
                                                                        self._options.k,
                                                                        self._options.unique,
                                                                        self._options.vocab_size,
                                                                        name='nce_sampling')
    target_score = tf.nn.sampled_softmax_loss(tf.transpose(self._output_weights),
                                              self._output_biases,
                                              ext_labels,
                                              hidden,
                                              self._options.k,
                                              self._options.vocab_size,
                                              num_true=1,
                                              sampled_values=(negative_samples,
                                                              true_expected_counts,
                                                              sampled_expected_counts),
                                              remove_accidental_hits=True,
                                              partition_strategy='mod',
                                              name='sampled_softmax_loss')
    loss = tf.reduce_mean(target_score, name='target_mean')
    if self._options.freqScores:
      losses = tfutils._compute_ranged_scores(target_score, labels, self._options.ranges)
      loss = [loss] + losses
    return loss

  def norm_loss(self, hidden, labels):
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.to_int64(tf.range(0, self._options.batch_size)), 1)
    concated = tf.concat(axis=1, values=[indices, labels])
    onehot_labels = tf.sparse_to_dense(
      concated, tf.to_int64(tf.stack([self._options.batch_size, self._options.vocab_size])), 1.0, 0.0)
    self._output = tf.matmul(hidden, self._output_weights) + self._output_biases
    scores = tf.exp(self._output)
    norm = tf.reduce_sum(scores, 1)
    cross_entropy = - tf.log(tf.reduce_sum(tf.multiply(scores, onehot_labels), 1) / norm)
    loss = tf.reduce_mean(cross_entropy) + self._options.alpha * tf.reduce_mean(tf.log(norm) ** 2)
    return loss

  def loss(self, hidden, labels):
    self._output = tf.matmul(hidden, self._output_weights) + self._output_biases
    ext_labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.to_int64(tf.range(0, self._options.batch_size)), 1)
    concated = tf.concat(axis=1, values=[indices, ext_labels])
    onehot_labels = tf.sparse_to_dense(
      concated, tf.to_int64(tf.stack([self._options.batch_size, self._options.vocab_size])), 1.0, 0.0)  
    scores = tf.exp(self._output)
    norms = tf.reduce_sum(scores, 1)
    cross_entropy = -tf.log(tf.reduce_sum(tf.multiply(scores, onehot_labels), 1) / norms)
    loss = tf.reduce_mean(cross_entropy)
    norm = tf.reduce_mean(norms)
    if self._options.freqScores:
      losses = tfutils._compute_ranged_scores(cross_entropy, labels, self._options.ranges)
      norms = tfutils._compute_ranged_scores(norms, labels, self._options.ranges)
      loss = [loss] + losses
      norm = [norm] + norms
    return loss, norm

  def optimize(self, loss):
    self._lr = self._options.learning_rate
    optimizer = tf.train.AdamOptimizer(self._lr)
    train = optimizer.minimize(loss + self._options.reg*tf.add_n(tf.get_collection('losses')),
                               global_step=self.global_step)
    self._train = train
    
  def build_graph(self):
    self._hidden = self.forward(self._examples, self._labels)
    if self._options.obj == 'nce' and self._training:
      loss, self._noNoiseLoss = self.nce_loss(self._hidden, self._labels)
      self._ent, self._norm = self.loss(self._hidden, self._labels)
    elif self._options.obj == 'blackOut' and self._training:
      loss, self._noNoiseLoss, self._noiseRatio = self.blackOut_loss(self._hidden, self._labels)
      self._ent, self._norm = self.loss(self._hidden, self._labels)
    elif self._options.obj == 'target' and self._training:
      loss = self.target_loss(self._hidden, self._labels)
      self._ent, self._norm = self.loss(self._hidden, self._labels)
    elif self._options.obj == 'norm' and self._training:
      loss = self.norm_loss(self._hidden, self._labels)
      self._ent, self._norm = self.loss(self._hidden, self._labels)
    else:
      loss, self._norm = self.loss(self._hidden, self._labels)
      self._ent = loss
    if self._training:
      self.optimize(loss)
    self._loss = loss
    self._monitored = [self._ent, self._norm]
    if (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
      self._monitored.append(self._noNoiseLoss)
    if self._options.obj == 'blackOut' and self._training:
      self._monitored.append(self._noiseRatio)

  def call(self, results_file = None):
    start_time = time.time()
    average_score = np.zeros(1 + self._options.nb_ranges)
    average_ent = np.zeros(1 + self._options.nb_ranges)
    average_norm = np.zeros(1 + self._options.nb_ranges)
    if (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
      average_score_noNoise = np.zeros(1 + self._options.nb_ranges)        
    if self._options.obj == 'blackOut' and self._training:
      average_ratio = np.zeros(1 + self._options.nb_ranges)

    if self._training:
      n_steps = self._options.n_training_steps // self._options.training_sub
      op = self._train
      display = n_steps // self._options.display_step
      call = "Training:"
    else:
      n_steps = self._options.n_testing_steps 
      op = tf.no_op()
      display = n_steps-1
      call = "Testing:"

    for step in xrange(n_steps):
      if (self._options.noise == 'bigram' and self._training):
        _, self._noiseDistrib = self._session.run([self._eval, self._meanNoiseDistrib])
      _, score, monitored = self._session.run([op, self._loss, self._monitored])

      # Record monitored values
      ent = monitored.pop(0)
      norm = monitored.pop(0)
      if (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
        scoreNoNoise = monitored.pop(0)
      if self._options.obj == 'blackOut' and self._training:
        scoreRatio = monitored.pop(0)
        
      average_score+= score
      average_ent+= ent
      average_norm+= norm
      if (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
        average_score_noNoise += scoreNoNoise
      if self._options.obj == 'blackOut' and self._training:
        average_ratio += scoreRatio

      if not self._options.freqScores:
        if self._training:
          if step % (display) == 0:
            print(" %s Perplexity, score and norm at batch %i : %.3f, %.3f, %.3f; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                                          ent, score,
                                                                                                                          np.log(norm), (time.time() - start_time) / (step + 1) ))
        else:
          if step % (display) == 0 and step > 0:
            print(" %s Perplexity, score and norm at batch %i : %.3f; %.3f, %.3f; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                                          np.exp(average_ent/(step+1)),
                                                                                                                          average_score/(step+1),
                                                                                                                          np.log(average_norm/(step+1)), (time.time() - start_time) / (step + 1) ))
      else:
        if self._training:
          if step % (display) == 0:
            print(np.exp(average_ent/(step+1)))
        else:
          if step % (display) == 0 and step > 0:
            print(np.exp(average_ent/(step+1)))

    #Last step: writing on file:
    end_time = time.time()
    if not results_file == None:
      results_file.write( str(average_ent/(step+1)) + '\n')
      results_file.write( str(average_score/(step+1)) + '\n')
      results_file.write( str(average_norm/(step+1)) + '\n')
      if (self._options.obj == 'nce' or self._options.obj == 'blackOut') and self._training:
        results_file.write( str(average_score_noNoise/(step+1)) + '\n')
      if self._options.obj == 'blackOut' and self._training:
        results_file.write( str(average_ratio/(step+1)) + '\n')
      results_file.write( str((end_time - start_time)/(step + 1)) + '\n')
      results_file.flush()
          
        
