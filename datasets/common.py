# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Basic libs
import numpy as np
import tensorflow as tf
import time

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

from utils.ply import read_ply

# Load custom operation
tf_neighbors_module = tf.load_op_library('tf_custom_ops/tf_neighbors.so')
tf_batch_neighbors_module = tf.load_op_library('tf_custom_ops/tf_batch_neighbors.so')
tf_subsampling_module = tf.load_op_library('tf_custom_ops/tf_subsampling.so')
tf_batch_subsampling_module = tf.load_op_library('tf_custom_ops/tf_batch_subsampling.so')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


def tf_batch_subsampling(points, batches_len, sampleDl):
    return tf_batch_subsampling_module.batch_grid_subsampling(points, batches_len, sampleDl)


def tf_batch_neighbors(queries, supports, q_batches, s_batches, radius):
    return tf_batch_neighbors_module.batch_ordered_neighbors(queries, supports, q_batches, s_batches, radius)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/


class Dataset:
    """
    Class managing data input for the network
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, name):
        # Name of the dataset
        self.name = name

        # Parameters for the files
        # ************************

        # Path of the folder containing ply files
        self.path = None

        # Parameters for the Labels (some attr are not used for completion task)
        # *************************

        # Number of different categories
        self.num_categories = 0

        # The different label values
        self.synset_values = []

        # The different category names
        self.category_names = []

        # Dict from synset to [0:num_categories] indices
        self.synset_to_idx = {}

        self.category_to_synset = {}

        # Dict from labels to names
        self.label_to_names = {}

        self.synset_to_category = {}

        # Other Parameters
        # ****************

        # Max number of convolution neighbors
        self.neighborhood_limits = None

        # List of ignored categories for this dataset
        self.ignored_categories = []

        # Type of task performed on this dataset
        self.network_model = ''

        # Number of threads used in input pipeline
        self.num_threads = 1

    def init_synsets(self):
        # Initiate all synset parameters given the synset_to_category dict
        self.num_categories = len(self.synset_to_category)
        self.synset_values = np.sort([k for k, v in self.synset_to_category.items()])
        self.category_names = [self.synset_to_category[k] for k in self.synset_values]
        self.synset_to_idx = {l: i for i, l in enumerate(self.synset_values)}
        self.category_to_synset = {v: k for k, v in self.synset_to_category.items()}

    # Data augmentation methods
    # ------------------------------------------------------------------------------------------------------------------

    def tf_augment_input(self, stacked_points, batch_inds, config):

        # Parameter
        num_batches = batch_inds[-1] + 1

        ##########
        # Rotation
        ##########

        if config.augment_rotation == 'vertical':

            # Choose a random angle for each element
            theta = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)

            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))

            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)

            # Apply rotations
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])

        elif config.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches,))

        else:
            raise ValueError('Unknown rotation augmentation : ' + config.augment_rotation)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = config.augment_scale_min
        max_s = config.augment_scale_max

        if config.augment_scale_anisotropic:
            s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((num_batches, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if config.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((num_batches, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.gather(s, batch_inds)

        # Apply scales
        stacked_points = stacked_points * stacked_scales

        #######
        # Noise
        #######

        noise = tf.random_normal(tf.shape(stacked_points), stddev=config.augment_noise)
        stacked_points = stacked_points + noise

        return stacked_points, s, R

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        return neighbors[:, :self.neighborhood_limits[layer]]

    def tf_get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """

        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):
            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1),
                                 lambda: tf.zeros((), dtype=tf.int32),
                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                                lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
                                lambda: tf.zeros((), dtype=tf.int32))

            # Update current element indices
            inds_before = tf.zeros((num_before,), dtype=tf.int32)
            inds_in = tf.fill((num_in,), batch_i)
            inds_after = tf.zeros((num_after,), dtype=tf.int32)
            n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

            b_inds += n_inds

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
                                                           tf.TensorShape([None])])

        return batch_inds

    def tf_stack_batch_inds(self, stacks_len):
        """

        :param stacks_len:
        :return: (Batch_num x max_num_pc) padded with the sum of all point cloud points constant
        """

        # Initiate batch inds tensor
        num_points = tf.reduce_sum(stacks_len)
        max_points = tf.reduce_max(stacks_len)
        batch_inds_0 = tf.zeros((0, max_points), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):
            # Create this element indices
            element_inds = tf.expand_dims(tf.range(point_i, point_i + stacks_len[batch_i]), axis=0)

            # Pad to right size
            padded_inds = tf.pad(element_inds,
                                 [[0, 0], [0, max_points - stacks_len[batch_i]]],
                                 "CONSTANT",
                                 constant_values=num_points)

            # Concatenate batch indices
            b_inds = tf.concat((b_inds, padded_inds), axis=0)

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        fixed_shapes = [tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None, None])]
        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=fixed_shapes)

        # Add a last column with shadow neighbor if there is not
        def f1(): return tf.pad(batch_inds, [[0, 0], [0, 1]], "CONSTANT", constant_values=num_points)

        def f2(): return batch_inds

        batch_inds = tf.cond(tf.equal(num_points, max_points * tf.shape(stacks_len)[0]), true_fn=f1, false_fn=f2)

        return batch_inds

    # Input pipeline methods
    # ------------------------------------------------------------------------------------------------------------------

    def init_input_pipeline(self, config):
        """
        Prepare the input pipeline with tf.Dataset class
        """

        ######################
        # Calibrate parameters
        ######################

        print('Initiating input pipelines')

        # Update num of categories in config
        config.num_categories = self.num_categories - len(self.ignored_categories)
        config.ignored_category_ids = [self.synset_to_idx[ign_label] for ign_label in self.ignored_categories]

        # Update network model in config from ds
        config.network_model = self.network_model

        # Calibrate generators to batch_num or use static batch limit
        if config.per_cloud_batch:
            self.batch_limit = config.batch_num
        else:
            self.batch_limit = self.calibrate_batches(config)

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = np.full(config.num_layers, hist_n, dtype=np.int32)

        # Calibrate max neighbors number for each layer
        self.calibrate_neighbors(config)

        ################################
        # Initiate tensorflow parameters
        ################################

        # Reset graph
        tf.reset_default_graph()

        # Set random seed (You also have to set it in network_architectures.weight_variable)
        # np.random.seed(42)
        # tf.set_random_seed(42)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen('train', config)
        gen_function_val, _, _ = self.get_batch_gen('valid', config)
        map_func = self.get_tf_mapping(config)

        ##################
        # Training dataset
        ##################

        # Create batched dataset from generator
        self.train_data = tf.data.Dataset.from_generator(gen_function,
                                                         gen_types,
                                                         gen_shapes)

        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        self.train_data = self.train_data.prefetch(10)

        ##############
        # Valid dataset
        ##############

        # Create batched dataset from generator
        self.val_data = tf.data.Dataset.from_generator(gen_function_val,
                                                       gen_types,
                                                       gen_shapes)

        # Transform inputs
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        self.val_data = self.val_data.prefetch(10)

        #################
        # Common iterator
        #################

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        self.flat_inputs = iter.get_next()

        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)

    def init_test_input_pipeline(self, config):
        """
        Prepare the input pipeline with tf.Dataset class
        """

        print('Initiating test input pipelines')

        ######################
        # Calibrate parameters
        ######################

        # Update num categories in config
        config.num_categories = self.num_categories - len(self.ignored_categories)
        config.ignored_category_ids = [self.synset_to_idx[ign_label] for ign_label in self.ignored_categories]

        # Update network model in config
        config.network_model = self.network_model

        # Calibrate generators to batch_num or use static batch limit
        if config.per_cloud_batch:
            self.batch_limit = config.batch_num
        else:
            self.batch_limit = self.calibrate_batches(config)

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = np.full(config.num_layers, hist_n, dtype=np.int32)

        # Calibrate max neighbors number
        self.calibrate_neighbors(config)

        ################################
        # Initiate tensorflow parameters
        ################################

        # Reset graph
        tf.reset_default_graph()

        # Set random seed (You also have to set it in network_architectures.weight_variable)
        # np.random.seed(42)
        # tf.set_random_seed(42)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen('test', config)
        map_func = self.get_tf_mapping(config)

        ##############
        # Test dataset
        ##############

        # Create batched dataset from generator
        self.test_data = tf.data.Dataset.from_generator(gen_function,
                                                        gen_types,
                                                        gen_shapes)

        self.test_data = self.test_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        self.test_data = self.test_data.prefetch(10)

        #################
        # Common iterator
        #################

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.test_data.output_types, self.test_data.output_shapes)
        self.flat_inputs = iter.get_next()

        # create the initialisation operations
        self.test_init_op = iter.make_initializer(self.test_data)

    def calibrate_batches(self, config):
        if 'cloud' in self.network_model:
            if len(self.input_trees['train']) > 0:
                split = 'train'
            else:
                split = 'test'

            N = (10000 // len(self.input_trees[split])) + 1
            sizes = []

            # Take a bunch of example neighborhoods in all clouds
            for i, tree in enumerate(self.input_trees[split]):
                # Randomly pick points
                points = np.array(tree.data, copy=False)
                rand_inds = np.random.choice(points.shape[0], size=N, replace=False)
                rand_points = points[rand_inds]
                noise = np.random.normal(scale=config.in_radius / 4, size=rand_points.shape)
                rand_points += noise.astype(rand_points.dtype)
                neighbors = tree.query_radius(points[rand_inds], r=config.in_radius)

                # Only save neighbors lengths
                sizes += [len(neighb) for neighb in neighbors]

            sizes = np.sort(sizes)

        else:
            if len(self.partial_points['train']) > 0:
                split = 'train'
            else:
                split = 'test'

            # Get sizes at training and sort them
            sizes = np.sort([p.shape[0] for p in self.partial_points[split]])

        # Higher bound for batch limit
        lim = sizes[-1] * config.batch_num

        # Biggest batch size with this limit
        sum_s = 0
        max_b = 0
        for i, s in enumerate(sizes):
            sum_s += s
            if sum_s > lim:
                max_b = i
                break

        # With a proportional corrector, find batch limit which gets the wanted batch_num
        estim_b = 0
        for i in range(10000):
            # Compute a random batch
            rand_shapes = np.random.choice(sizes, size=max_b, replace=False)
            b = np.sum(np.cumsum(rand_shapes) < lim)

            # Update estim_b (low pass filter instead of real mean
            estim_b += (b - estim_b) / min(i + 1, 100)

            # Correct batch limit
            lim += 10.0 * (config.batch_num - estim_b)

        return lim

    def calibrate_neighbors(self, config, keep_ratio=0.8, samples_threshold=10000):

        # Create a tensorflow input pipeline
        # **********************************
        if 'cloud' in self.network_model:
            if len(self.input_trees['train']) > 0:
                split = 'train'
            else:
                split = 'test'
        else:
            if len(self.partial_points['train']) > 0:
                split = 'train'
            else:
                split = 'test'

        # Get mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen(split, config)
        map_func = self.get_tf_mapping(config)

        # Create batched dataset from generator
        train_data = tf.data.Dataset.from_generator(gen_function,
                                                    gen_types,
                                                    gen_shapes)

        train_data = train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        train_data = train_data.prefetch(10)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        flat_inputs = iter.get_next()

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_data)

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Create a local session for the calibration.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        with tf.Session(config=cProto) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Initialise iterator with train data
            sess.run(train_init_op)

            # Get histogram of neighborhood sizes in 1 epoch max
            # **************************************************

            neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
            t0 = time.time()
            mean_dt = np.zeros(2)
            last_display = t0
            epoch = 0
            training_step = 0
            while epoch < 1 and np.min(np.sum(neighb_hists, axis=1)) < samples_threshold:
                try:
                    # Get next inputs
                    t = [time.time()]
                    ops = flat_inputs[config.num_layers:2 * config.num_layers]  # neighbor_mat is 5th index of flat in
                    neighbors = sess.run(ops)
                    t += [time.time()]

                    # Update histogram (neighb_mat.shape[0] are shadow neighbors)
                    counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)
                    t += [time.time()]

                    # Average timing
                    mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Console display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Calib Neighbors {:08d} : timings {:4.2f} {:4.2f}'
                        print(message.format(training_step,
                                             1000 * mean_dt[0],
                                             1000 * mean_dt[1]))

                    training_step += 1

                except tf.errors.OutOfRangeError:
                    print('End of train dataset')
                    epoch += 1

            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

            self.neighborhood_limits = percentiles
            print('\n')

        return

    def get_batch_gen(self, split, config):
        raise ValueError('You need to implement a "get_batch_gen" method for this dataset.')

    def get_tf_mapping(self, config):
        raise ValueError('You need to implement a "get_tf_mapping" method for this dataset.')

    def tf_completion_inputs(self,
                             config,
                             stacked_points,
                             stacked_features,
                             stacked_complete,
                             stacks_lengths,
                             batch_inds):

        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keep_dims=True)  # min partial pc size in batch
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)  # smaller pc-->bigger weight
        stacked_weights = tf.gather(batch_weights, batch_inds)

        # Starting radius of convolutions
        r_normal = config.first_subsampling_dl * config.KP_extent * 2.5  # check KPConv paper --> Net params

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_batches_len = []

        # When using per cloud batch generator, points need to be reshaped
        # if config.per_cloud_batch:
        #     stacked_points = tf.reshape(stacked_points, [-1, 3])

        ######################
        # Loop over the blocks
        ######################
        for block_i, block in enumerate(config.architecture):

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * config.density_parameter / (config.KP_extent * 2.5)
                else:
                    r = r_normal
                conv_i = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            else:
                # This layer only perform pooling, no neighbors required
                conv_i = tf.zeros((0, 1), dtype=tf.int32)

            # Pooling neighbors indices
            # *************************
            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:
                # New subsampling length (dl doubled)
                dl = 2 * r_normal / (config.KP_extent * 2.5)

                # Subsampled points
                pool_p, pool_b = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * config.density_parameter / (config.KP_extent * 2.5)
                else:
                    r = r_normal

                # Subsample indices (get closest un-pooled neighbors indices for each pooled point)
                pool_i = tf_batch_neighbors(pool_p, stacked_points, pool_b, stacks_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = tf_batch_neighbors(stacked_points, pool_p, stacks_lengths, pool_b, 2 * r)
            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = tf.zeros((0, 1), dtype=tf.int32)
                pool_p = tf.zeros((0, 3), dtype=tf.float32)
                pool_b = tf.zeros((0,), dtype=tf.int32)
                up_i = tf.zeros((0, 1), dtype=tf.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            up_i = self.big_neighborhood_filter(up_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i]
            input_pools += [pool_i]
            input_upsamples += [up_i]
            input_batches_len += [stacks_lengths]

            # New pooled points for next layer
            stacked_points = pool_p
            stacks_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

        ###############
        # Return inputs
        ###############

        # Batch unstacking (with last layer indices for optional classification loss)
        stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])

        # Batch unstacking (with last layer indices for optional classification loss)
        stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples
        li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
        li += [stacked_complete]

        return li
