import sys
import os
import math
import time

import numpy as np
import tensorflow as tf
import tfutils

#Define the class for the Language Model
class LM(object):

  def __init__(self, options, session, inputs, training=True):
    self._options = options
    self._session = session
    self._training = training
    if (self._options.noise == 'bigram' and self._training):
      self._examples, self._labels, self._currentNoiseDistrib = inputs
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
                                                                        unigrams=self._options.noiseDistrib,
                                                                        name='nce_sampling')
      logits, labels_nce = tf.nn._compute_sampled_logits(tf.transpose(self._output_weights),
                                                         self._output_biases,
                                                         hidden,
                                                         labels_ext,
                                                         self._options.k,
                                                         self._options.vocab_size,
                                                         num_true=1,
                                                         sampled_values= (negative_samples,
                                                                          true_expected_counts,
                                                                          sampled_expected_counts),
                                                         subtract_log_q= True,
                                                         remove_accidental_hits = True,
                                                         name='nce_loss_1')
      sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels_nce, name="nce_loss_2")
      positive_score = sampled_losses[:,0]
      true_loss = tf.reduce_mean(positive_score, name='true_mean')
    #Different noise samples for each batch
    else:
      (negative_samples,
       true_expected_counts,
       sampled_expected_counts) = tf.nn.fixed_unigram_candidate_sampler(labels_ext,
                                                                        1,
                                                                        self._options.k * self._options.batch_size,
                                                                        self._options.unique,
                                                                        self._options.vocab_size,
                                                                        distortion=self._options.distortion,
                                                                        num_reserved_ids=0,
                                                                        unigrams=self._options.noiseDistrib,
                                                                        name='nce_sampling')
      true_expected_counts = tf.scalar_mul(1.0 / self._options.batch_size, true_expected_counts)
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
                                                                    remove_accidental_hits = True,
                                                                    name='nce_loss_1_batched')      
      sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels_nce, name="nce_loss_2_batched")
      positive_score = sampled_losses[:,0]
      true_loss = tf.reduce_mean(positive_score, name='true_mean')
    nce_score = tf.nn._sum_rows(sampled_losses)
    loss = tf.reduce_mean(nce_score, name='nce_mean')
    return loss, true_loss

  def contDepNce_loss(self, hidden, labels, currentNoiseDistrib):
    labels_ext = tf.expand_dims(labels, 1)
    #Same noise samples for the whole batch: we can compute nce_loss by batch with the usual function
    #Impossible for contexte dependant noise
    if self._options.batchedNoise:
      print "Error: Bigram can't work with batched noise - noise is context-dependent!"
      sys.exit()
    else:
      proba = tf.nn.log_softmax(tf.log(tf.cast(currentNoiseDistrib, dtype='float32')))
      samples = tf.multinomial(tf.log(tf.cast(currentNoiseDistrib, dtype='float32')),
                               self._options.k,
                               name = 'nce_sampling')        
      print(samples.get_shape())
      negative_samples = tf.reshape(samples, [self._options.k * self._options.batch_size])              

      labels_indexes = tf.transpose(tf.pack([tf.cast(tf.range(self._options.batch_size), dtype='int64'), labels]))
      idx = tf.cast(tf.range(self._options.batch_size), dtype='int64')
      idx = tf.reshape(idx, [-1, 1])    
      idx = tf.tile(idx, [1, self._options.k])  
      idx = tf.reshape(idx, [-1])       
      sampled_indexes = tf.transpose(tf.pack([idx, negative_samples]))
      true_expected_counts = tf.scalar_mul(self._options.k, tf.exp(tf.expand_dims(tf.gather_nd(proba, labels_indexes), 1)))
      sampled_expected_counts = tf.scalar_mul(self._options.k, tf.exp(tf.gather_nd(proba, sampled_indexes)))
      
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
                                                                       remove_accidental_hits = True,
                                                                       name='nce_loss_1_batched')

      sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels_nce, name="nce_loss_2_batched")
      positive_score = sampled_losses[:,0]
      true_loss = tf.reduce_mean(positive_score, name='true_mean')
      
      nce_score = tf.nn._sum_rows(sampled_losses)
      loss = tf.reduce_mean(nce_score, name='nce_mean')                                                             
      return loss, true_loss

  def target_loss(self, hidden, labels):
    labels = tf.expand_dims(labels, 1)

    (negative_samples,
     true_expected_counts,
     sampled_expected_counts) = tf.nn.learned_unigram_candidate_sampler(labels,
                                                                      1,
                                                                      self._options.k,
                                                                      self._options.unique,
                                                                      self._options.vocab_size,
                                                                      name='nce_sampling')
    
    target_score = tf.nn.sampled_softmax_loss(tf.transpose(self._output_weights),
                                              self._output_biases,
                                              hidden,
                                              labels,
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

    return loss

  def norm_loss(self, hidden, labels):
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.to_int64(tf.range(0, self._options.batch_size)), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
      concated, tf.to_int64(tf.pack([self._options.batch_size, self._options.vocab_size])), 1.0, 0.0)
    self._output = tf.matmul(hidden, self._output_weights) + self._output_biases
    scores = tf.exp(self._output)
    norm = tf.reduce_sum(scores, 1)
    cross_entropy = - tf.log(tf.reduce_sum(tf.mul(scores, onehot_labels), 1) / norm)
    loss = tf.reduce_mean(cross_entropy) + self._options.alpha * tf.reduce_mean(tf.log(norm) ** 2)
    return loss

  def loss(self, hidden, labels):
    self._output = tf.matmul(hidden, self._output_weights) + self._output_biases
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.to_int64(tf.range(0, self._options.batch_size)), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
      concated, tf.to_int64(tf.pack([self._options.batch_size, self._options.vocab_size])), 1.0, 0.0)  
    scores = tf.exp(self._output)
    norms = tf.reduce_sum(scores, 1)
    cross_entropy = -tf.log(tf.reduce_sum(tf.mul(scores, onehot_labels), 1) / norms)
    loss = tf.reduce_mean(cross_entropy)
    norm = tf.reduce_mean(norms)
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
      if self._options.noise == 'bigram':
        loss, self._noNoiseLoss = self.contDepNce_loss(self._hidden, self._labels, self._currentNoiseDistrib)
      else:
        loss, self._noNoiseLoss = self.nce_loss(self._hidden, self._labels)
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
    if self._options.obj == 'nce' and self._training:
      self._monitored.append(self._noNoiseLoss)

  def call(self, results_file = None):
    start_time = time.time()
    average_score = 0.0
    average_ent = 0.0
    average_norm = 0.0
    if self._options.obj == 'nce' and self._training:
      average_score_noNoise = 0.0        
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
      _, score, monitored = self._session.run([op, self._loss, self._monitored])

      # Record monitored values
      average_score+= score
      ent = monitored.pop(0)
      norm = monitored.pop(0)
      average_ent+= ent
      average_norm+= norm
      if self._options.obj == 'nce' and self._training:
        scoreNoNoise = monitored.pop(0)
        average_score_noNoise += scoreNoNoise

      if self._training:
        if step % (display) == 0:
          print(" %s Perplexity, score and norm at batch %i : %.3f, %.3f, %.3f; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                                        ent, score,
                                                                                                                        math.log(norm), (time.time() - start_time) / (step + 1) ))
      else:
        if step % (display) == 0 and step > 0:
          #TODO: save values for testing and plot
          print(" %s Perplexity, score and norm at batch %i : %.3f; %.3f, %.3f; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                                        math.exp(average_ent/(step+1)),
                                                                                                                        average_score/(step+1),
                                                                                                                        math.log(average_norm/(step+1)), (time.time() - start_time) / (step + 1) ))
      #Last step: writing on file:
      if step == (n_steps - 1):
        if not results_file == None:
          results_file.write( str(average_ent/(step+1)) + '\n')
          results_file.write( str(average_score/(step+1)) + '\n')
          results_file.write( str(average_norm/(step+1)) + '\n')
          if self._options.obj == 'nce' and self._training:
            results_file.write( str(average_score_noNoise/(step+1)) + '\n')
          results_file.flush()

        
