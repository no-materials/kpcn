# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import numpy as np
import tensorflow as tf

import kernels.convolution_ops as conv_ops
from utils.metrics import chamfer, earth_mover


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[-1]))
    initial = tf.round(initial * tf.constant(1000, dtype=tf.float32)) / tf.constant(1000, dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name='bias')


def ind_max_pool(x, inds):
    """
    This tensorflow operation compute a maxpooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.reduce_min(x, axis=0, keep_dims=True)], axis=0)

    # Get features for each pooling cell [n2, max_num, d]
    pool_features = tf.gather(x, inds, axis=0)

    # Pool the maximum
    return tf.reduce_max(pool_features, axis=1)


def closest_pool(x, inds):
    """
    This tensorflow operation compute a pooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)

    # Get features for each pooling cell [n2, d]
    pool_features = tf.gather(x, inds[:, 0], axis=0)

    return pool_features


def KPConv(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv(query_points,
                           support_points,
                           neighbors_indices,
                           features,
                           K_values,
                           fixed=config.fixed_kernel_points,
                           KP_extent=extent,
                           KP_influence=config.KP_influence,
                           aggregation_mode=config.convolution_mode, )


def KPConv_deformable(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a deformable KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv_deformable(query_points,
                                      support_points,
                                      neighbors_indices,
                                      features,
                                      K_values,
                                      fixed=config.fixed_kernel_points,
                                      KP_extent=extent,
                                      KP_influence=config.KP_influence,
                                      aggregation_mode=config.convolution_mode,
                                      modulated=config.modulated)


def batch_norm(x, use_batch_norm=True, momentum=0.99, training=True):
    """
    This tensorflow operation compute a batch normalization.
    > x = [n1, d] features matrix
    >> output = [n1, d] normalized, scaled, offset features matrix
    """

    if use_batch_norm:
        return tf.layers.batch_normalization(x,
                                             momentum=momentum,
                                             epsilon=1e-6,
                                             training=training)

    else:
        # Just add biases
        beta = tf.Variable(tf.zeros([x.shape[-1]]), name='offset')
        return x + beta


def leaky_relu(features, alpha=0.2):
    return tf.nn.leaky_relu(features, alpha=alpha, name=None)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Convolution blocks
#       \************************/
#

def unary_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """

    w = weight_variable([int(features.shape[1]), fdim])
    x = conv_ops.unary_convolution(features, w)
    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def simple_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind],
               inputs['points'][layer_ind],
               inputs['neighbors'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def resnetb_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)  # [n_points, 2*f_dim = 128]


def resnetb_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind + 1],
                   inputs['points'][layer_ind],
                   inputs['pools'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def resnetb_deformable_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconvDef > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv_deformable(inputs['points'][layer_ind],
                              inputs['points'][layer_ind],
                              inputs['neighbors'][layer_ind],
                              x,
                              w,
                              radius,
                              config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_deformable_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv_deformable(inputs['points'][layer_ind + 1],
                              inputs['points'][layer_ind],
                              inputs['pools'][layer_ind],
                              x,
                              w,
                              radius,
                              config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def nearest_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an upsampling by nearest interpolation
    """

    with tf.variable_scope('nearest_upsample'):
        upsampled_features = closest_pool(features, inputs['upsamples'][layer_ind - 1])

    return upsampled_features


def get_block_ops(block_name):
    if block_name == 'unary':
        return unary_block

    if block_name == 'simple':
        return simple_block

    if block_name == 'simple_strided':
        return simple_strided_block

    elif block_name == 'resnet':
        return resnet_block

    elif block_name == 'resnetb':
        return resnetb_block

    elif block_name == 'resnetb_light':
        return resnetb_light_block

    elif block_name == 'resnetb_deformable':
        return resnetb_deformable_block

    elif block_name == 'inception_deformable':
        return inception_deformable_block()

    elif block_name == 'resnetb_strided':
        return resnetb_strided_block

    elif block_name == 'resnetb_light_strided':
        return resnetb_light_strided_block

    elif block_name == 'resnetb_deformable_strided':
        return resnetb_deformable_strided_block

    elif block_name == 'inception_deformable_strided':
        return inception_deformable_strided_block()

    elif block_name == 'vgg':
        return vgg_block

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return max_pool_block

    elif block_name == 'global_average':
        return global_average_block

    elif block_name == 'nearest_upsample':
        return nearest_upsample_block

    elif block_name == 'simple_upsample':
        return simple_upsample_block

    elif block_name == 'resnetb_upsample':
        return resnetb_upsample_block

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Architectures
#       \*******************/
#

def assemble_CNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    This assembles the 'encoder' part of the KFCNN of KPCONV paper
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # Current radius of convolution and feature dimension
    r = config.first_subsampling_dl * config.density_parameter
    layer = 0
    fdim = config.first_features_dim

    # Input features
    features = inputs['features']
    F = []

    # Boolean of training
    training = dropout_prob < 0.99

    # Loop over consecutive blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture):

        # Detect change to next layer
        if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
            # Save this layer features
            F += [features]

        # Detect upsampling block to stop
        if 'upsample' in block:
            break

        with tf.variable_scope('layer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):
            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'pool' in block or 'strided' in block:
            # Update radius and feature dimension for next layer
            layer += 1
            r *= 2
            fdim *= 2
            block_in_layer = 0

        # Save feature vector after global pooling
        if 'global' in block:
            # Save this layer features
            F += [features]

    return F


def assemble_KPCN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer  # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:
            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, F[layer]), axis=1)

    return features


def completion_head(features, config, dropout_prob):
    # Boolean of training
    training = dropout_prob < 0.99

    # Fully connected layer2
    with tf.variable_scope('fc1'):
        w = weight_variable([int(features.shape[1]), 1024])
        features = leaky_relu(batch_norm(tf.matmul(features, w),
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))
    with tf.variable_scope('fc2'):
        w = weight_variable([1024, config.num_coarse * 3])
        features = leaky_relu(batch_norm(tf.matmul(features, w),
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    return tf.reshape(features, [-1, config.num_coarse, 3])


# TODO: change this to Chamfer Distance & EMD - add fine...foldingnet...
# TODO: add dynamic alpha tf variable
def completion_loss(coarse, inputs, config, alpha, batch_average=False):
    # print(inputs['complete_points'].shape)
    # print(coarse.shape)

    # TODO: for compl sizes in.. subst config.num_coarse slice to get true coarsed gt and not a random 1024 points
    gt_ds = tf.reshape(inputs['complete_points'], [-1, config.num_coarse, 3])
    # print(gt_ds.shape)
    loss_coarse = earth_mover(coarse, gt_ds)

    # loss_fine = chamfer(fine, gt)
    #
    # loss = loss_coarse + alpha * loss_fine

    return loss_coarse
