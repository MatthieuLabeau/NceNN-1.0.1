from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

from tensorflow.python.ops.nn import _sum_rows

# pylint: enable=wildcard-import
def _compute_sampled_logits_by_batch(weights,
                                     biases,
                                     inputs,
                                     labels,
                                     num_sampled,
                                     k,
                                     num_classes,
                                     num_true,
                                     sampled_values,
                                     subtract_log_q=True,
                                     remove_accidental_hits=False,
                                     partition_strategy="mod",
                                     name=None):
    
    """
    Modification of _compute_sampled_logits able to use different noise
    samples for different true examples in the batch - used for context
    dependent noise samples.
    """
    
    if not isinstance(weights, list):
        weights = [weights]
        
    with ops.op_scope(weights + [biases, inputs, labels], name,
                      "compute_sampled_logits"):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

        # NOTE: pylint cannot tell that 'sampled_values' is a sequence                                                                                                                                
        # pylint: disable=unpacking-non-sequence                                                                                                                                              
        sampled, true_expected_count, sampled_expected_count = sampled_values
        # pylint: enable=unpacking-non-sequence                                                                                                                                            

        # labels_flat is a [batch_size * num_true] tensor                                                                                                                                          
        # sampled is a [num_sampled] int tensor                                                                                                                                                        
        all_ids = array_ops.concat(0, [labels_flat, sampled])

         # weights shape is [num_classes, dim]                                                                                                                                                     
        all_w = embedding_ops.embedding_lookup(
            weights, all_ids, partition_strategy=partition_strategy)
        all_b = embedding_ops.embedding_lookup(biases, all_ids)
        # true_w shape is [batch_size * num_true, dim]                                                                                                                                       
        # true_b is a [batch_size * num_true] tensor                                                                                                                                             
        true_w = array_ops.slice(
            all_w, [0, 0], array_ops.pack([array_ops.shape(labels_flat)[0], -1]))
        true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
        
        # inputs shape is [batch_size, dim]                                                                                                                                                      
        # true_w shape is [batch_size * num_true, dim]                                                                                                                                               
        # row_wise_dots is [batch_size, num_true, dim]                                                                                                                                             
        dim = array_ops.shape(true_w)[1:2]
        new_true_w_shape = array_ops.concat(0, [[-1, num_true], dim])
        row_wise_dots = math_ops.mul(
            array_ops.expand_dims(inputs, 1),
            array_ops.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields 
        
        # [batch_size, num_true] tensor of true_logits.                                                                                                                                           
        dots_as_matrix = array_ops.reshape(row_wise_dots,
                                           array_ops.concat(0, [[-1], dim]))
        true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = array_ops.reshape(true_b, [-1, num_true])
        true_logits += true_b
        
        # Lookup weights and biases for sampled labels.                                                                                                                                            
        #   sampled_w shape is [num_sampled, dim]                                                                                                                                                      
        #   sampled_b is a [num_sampled] float tensor                                                                                                                                             
        sampled_w = array_ops.slice(
            all_w, array_ops.pack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
        sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # sampled_b has shape [num_sampled]
        # Apply X*W'+B, which yields [batch_size, num_sampled]
        """
        Insert batch_matmul instead of matmul here:
        """
        batch_size = array_ops.shape(inputs)[0]
        # inputs from [batch_size, dim] to [batch_size, 1, dim]
        inputs = array_ops.expand_dims(inputs, 1)
        # sampled_w from [num_sampled = k * batch_size, dim] to [batch_size, dim, k]
        sampled_w = array_ops.reshape(
            sampled_w, array_ops.pack([batch_size, k, -1]))
        sampled_w = array_ops.transpose(sampled_w, perm=[0, 2, 1])
        # sampled_b from [num_sampled] to [batch_size, k]
        sampled_b = array_ops.reshape(
            sampled_b, array_ops.pack([batch_size, k]))
    
        # batch_matmul yields [batch_size, 1, k] that we reduce to [batch_size, k] and add sampled_b
        sampled_logits = array_ops.squeeze(
            math_ops.batch_matmul(
                inputs, sampled_w)
            ) + sampled_b
        
        """
        sampled_logits = math_ops.matmul(
        inputs, sampled_w, transpose_b=True) + sampled_b
        """
        """
        if remove_accidental_hits:               
            # Not implemented      
            acc_hits = candidate_sampling_ops.compute_accidental_hits(
                labels, sampled, num_true=num_true)
            acc_indices, acc_ids, acc_weights = acc_hits
        
            # This is how SparseToDense expects the indices.
            acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
            acc_ids_2d_int32 = array_ops.reshape(
                math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
            sparse_indices = array_ops.concat_v2([acc_indices_2d, acc_ids_2d_int32],
                                                 1, "sparse_indices")
            # Create sampled_logits_shape = [batch_size, num_sampled]
            sampled_logits_shape = array_ops.concat_v2(
                [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)],
                0)
            if sampled_logits.dtype != acc_weights.dtype:
                acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
                sampled_logits += sparse_ops.sparse_to_dense(
                    sparse_indices,
                    sampled_logits_shape,
                    acc_weights,
                    default_value=0.0,
                    validate_indices=False)
        """            
        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= math_ops.log(true_expected_count)
            """
            Here, reshape sampled_expected_count to be consistent with sampled_logits
            """
            sampled_expected_count = array_ops.reshape(
                sampled_expected_count, array_ops.pack([batch_size, k]))
            sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat(1, [true_logits, sampled_logits])
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat(1, [
                array_ops.ones_like(true_logits) / num_true,
                array_ops.zeros_like(sampled_logits)
                ])
    
    return out_logits, out_labels
