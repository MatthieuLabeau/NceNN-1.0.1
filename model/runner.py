# From : https://indico.io/blog/tensorflow-data-input-part2-extensions/
## ====================================================================

import tensorflow as tf
import threading
import numpy as np

class datarunner(object):
    def __init__(self, n, noise=False, vocab_size=None):
        self.examples_ph = tf.placeholder(dtype=tf.int64, shape=[None, n])
        self.labels_ph = tf.placeholder(dtype=tf.int64, shape=[None, ])
        self.noise = noise
        if noise:
            self.noise_ph = tf.placeholder(dtype=tf.int64, shape=[None, vocab_size])
            self.queue = tf.RandomShuffleQueue(shapes=[[n], [], [vocab_size]],
                                               dtypes=[tf.int64, tf.int64, tf.int64],
                                               capacity=2000,
                                               min_after_dequeue=1000)
            self.enqueue_op = self.queue.enqueue_many([self.examples_ph, self.labels_ph, self.noise_ph])
        else:
            self.queue = tf.RandomShuffleQueue(shapes=[[n], []],
                                               dtypes=[tf.int64, tf.int64],
                                               capacity=2000,
                                               min_after_dequeue=1000)
            self.enqueue_op = self.queue.enqueue_many([self.examples_ph, self.labels_ph])

    # Try to to preprocessing noise from a sparse matrix, doesn't work with queue though
    """
    def getCurrentNoiseDistrib(self, examples):
        out, _ = tf.listdiff(self.sparse.indices[:,0], examples[:,-1])
        _, indices_to_retain = tf.listdiff(self.sparse.indices[:,0], out)
        retain_values = tf.fill(tf.shape(indices_to_retain), True), bool_retain = tf.sparse_to_dense(indices_to_retain, tf.shape(self.sparse.indices[:,0]), retain_values, False)
        return tf.sparse_tensor_to_dense(tf.sparse_retain(self.sparse, bool_retain), default_value=0) + 1
    """

    def get_inputs(self, batch_size):
        return self.queue.dequeue_many(batch_size)

    def thread_main(self, session, sampler):
        if self.noise:
            for examples_batch, labels_batch, noise_batch in sampler:
                session.run(self.enqueue_op, feed_dict={self.examples_ph:examples_batch, self.labels_ph:labels_batch, self.noise_ph:noise_batch})
        else:
            for examples_batch, labels_batch in sampler:
                session.run(self.enqueue_op, feed_dict={self.examples_ph:examples_batch, self.labels_ph:labels_batch})


    def start_threads(self, session, sampler, n_threads=1):
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(session, sampler))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads

