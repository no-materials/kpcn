import tensorflow as tf
import numpy as np
import time
import json
import pickle
from sklearn.neighbors import KDTree

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# PLY IO
from utils.ply import read_ply, write_ply

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir, realpath, dirname

# Dataset parent class
from datasets.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

from utils.data import load_csv, load_h5, pad_cloudN

from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \**********************/
#


class ShapeNetBenchmark2048Dataset(Dataset):
    """
    ShapeNetBenchmark2048 dataset for completion task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, batch_num, input_pts, dataset_path, input_threads=8):
        """
        Initiation method.
        """
        Dataset.__init__(self, 'pc_shapenetCompletionBenchmark2048')

        self.synset_to_category = {
            '02691156': 'Airplane',
            '02958343': 'Car',
            '03001627': 'Chair',
            '03636649': 'Lamp',
            '04379243': 'Table',
            '02933112': 'Cabinet',
            '04256520': 'Sofa',
            '04530566': 'Boat',
        }

        self.init_synsets()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Number of models
        self.network_model = 'completion'

        # Partial, complete point clouds & categories used for each split
        self.partial_points = {}
        self.complete_points = {}
        self.ids = {}

        ##########################
        # Parameters for the files
        ##########################

        # Path of the dataset src folder
        self.dataset_path = dataset_path

        # Path to preprocessed data folder
        # self.data_path = join(dirname(dirname(realpath(__file__))),
        #                       'data',
        #                       'shapenetBenchmark2048')
        # if not exists(self.data_path):
        #     makedirs(self.data_path)

        self.batch_num = batch_num

        # Number of threads
        self.num_threads = input_threads

        self.input_pts = input_pts

        # Load classmaps
        classmap = load_csv(join(self.dataset_path, 'synsetoffset2category.txt'))
        self.classmap = {}
        for i in range(classmap.shape[0]):
            self.classmap[str(classmap[i][1]).zfill(8)] = classmap[i][0]

        # Split file paths lists
        self.train_split_file = join(self.dataset_path, 'train.list')
        self.valid_split_file = join(self.dataset_path, 'val.list')
        self.test_split_file = join(self.dataset_path, 'test.list')

        # Split data paths
        self.train_data_paths = sorted([join(self.dataset_path, 'train', 'partial', k.rstrip() + '.h5') for k in
                                        open(self.train_split_file).readlines()])
        self.val_data_paths = sorted([join(self.dataset_path, 'val', 'partial', k.rstrip() + '.h5') for k in
                                      open(self.valid_split_file).readlines()])
        self.test_data_paths = sorted([join(self.dataset_path, 'test', 'partial', k.rstrip() + '.h5') for k in
                                       open(self.test_split_file).readlines()])

        # make datasets dividable by batch num and set num of splits
        self.num_train = int(len(self.train_data_paths) / batch_num) * batch_num  # 28974
        self.train_data_paths = self.train_data_paths[0:self.num_train]
        self.num_valid = int(len(self.val_data_paths) / batch_num) * batch_num  # 800
        self.val_data_paths = self.val_data_paths[0:self.num_valid]
        self.num_test = int(len(self.test_data_paths) / batch_num) * batch_num  # 1184
        self.test_data_paths = self.test_data_paths[0:self.num_test]

    def get_pair(self, fname, train):
        partial = load_h5(fname)
        if train == 'test':
            gtpts = partial
            # gtpts = load_h5(fname.replace('partial', 'gt'))
        else:
            gtpts = load_h5(fname.replace('partial', 'gt'))
        # if train:
        #     gtpts, partial = augment_cloud([gtpts, partial], args)
        # partial = pad_cloudN(partial, 2048)
        return partial, gtpts

    def load_data(self, fname, split):
        pair = self.get_pair(fname, train=split)
        partial = pair[0]
        target = pair[1]
        cloud_meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]), 0), ]
        return target, cloud_meta, partial

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        self.partial_points = {'train': [], 'valid': [], 'test': []}
        self.complete_points = {'train': [], 'valid': [], 'test': []}
        self.ids = {'train': [], 'valid': [], 'test': []}

        split_list = ['train', 'valid', 'test']

        for split_type in split_list:

            # Restart timer
            t0 = time.time()

            # Load wanted points if possible
            print('\nLoading %s points' % split_type)
            filename = join(self.dataset_path, '{0:s}_{1:.3f}_record.pkl'.format(split_type, subsampling_parameter))

            if exists(filename):
                with open(filename, 'rb') as file:
                    self.partial_points[split_type], \
                    self.complete_points[split_type], \
                    self.ids[split_type] = pickle.load(file)

            # Else compute them from original points
            else:
                if split_type == 'train':
                    paths = self.train_data_paths
                elif split_type == 'valid':
                    paths = self.val_data_paths
                else:
                    paths = self.test_data_paths

                for file_iter, file_path in enumerate(paths):
                    # Call loading functions
                    print(file_path)
                    data = self.load_data(file_path, split_type)

                    if subsampling_parameter > 0:
                        sub_partial_points = grid_subsampling(data[2].astype(np.float32),
                                                              sampleDl=subsampling_parameter)
                        # padded_sub_partial = pad_cloudN(sub_partial_points, self.input_pts)
                        self.partial_points[split_type] += [sub_partial_points]
                        self.complete_points[split_type] += [data[0]]
                        self.ids[split_type] += [data[1]]
                        # plot_pcds(None, [data[2], sub_partial_points], ['partial', 'gt'], use_color=[0, 0], color=[None, None])

                    else:
                        # padded_partial = pad_cloudN(data[2], self.input_pts)
                        self.partial_points[split_type] += [data[2]]
                        self.complete_points[split_type] += [data[0]]
                        self.ids[split_type] += [data[1]]
                        # plot_pcds(None, [data[2], data[0]], ['partial', 'gt'], use_color=[0, 0], color=[None, None])

                # Save split pickle for later use
                with open(filename, 'wb') as file:
                    pickle.dump((self.partial_points[split_type],
                                 self.complete_points[split_type],
                                 self.ids[split_type]), file)

            lengths = [p.shape[0] for p in self.partial_points[split_type]]
            lengths.extend([p.shape[0] for p in self.complete_points[split_type]])
            sizes = [l * 4 * 3 for l in lengths]
            print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------
    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "train", "valid" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        # Balance training sample classes
        balanced = False

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}

        # Reset potentials
        print('OOOOOOOIIIIIIIIIII--------')
        print(split)
        print(len(self.ids[split]))
        self.potentials[split] = np.random.rand(len(self.ids[split])) * 1e-3

        ################
        # Def generators
        ################
        def dynamic_batch_point_based_gen():

            # Initiate concatenation lists
            tpp_list = []  # partial points
            tcp_list = []  # complete points
            tid_list = []  # ids
            ti_list = []  # cloud index
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'train':
                if balanced:
                    pick_n = int(np.ceil(self.num_train / self.num_classes))
                    gen_indices = []
                    for l in self.label_values:
                        label_inds = np.where(np.equal(self.input_labels[split], l))[0]
                        rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                        gen_indices += [rand_inds]
                    gen_indices = np.random.permutation(np.hstack(gen_indices))
                else:
                    gen_indices = np.random.permutation(self.num_train)

            elif split == 'valid':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            elif split == 'test':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            for p_i in gen_indices:

                # Get points
                new_partial_points = self.partial_points[split][p_i].astype(np.float32)
                new_complete_points = self.complete_points[split][p_i].astype(np.float32)
                n = new_partial_points.shape[0]  # num of points of selected partial point cloud

                # Collect labels
                input_category = self.ids[split][p_i]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tpp_list, axis=0),
                           np.concatenate(tcp_list, axis=0),
                           np.array(tid_list),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tpp_list]),
                           np.array([tc.shape[0] for tc in tcp_list]))
                    tpp_list = []
                    tcp_list = []
                    tid_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tpp_list += [new_partial_points]
                tcp_list += [new_complete_points]
                tid_list += [input_category[0]]
                ti_list += [p_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tpp_list, axis=0),
                   np.concatenate(tcp_list, axis=0),
                   np.array(tid_list),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tpp_list]),
                   np.array([tc.shape[0] for tc in tcp_list]))

        def static_batch_cloud_based_gen():

            # Initiate concatenation lists
            tpp_list = []  # partial points
            tcp_list = []  # complete points
            tid_list = []  # categories
            ti_list = []  # cloud index
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'train':
                if balanced:
                    pick_n = int(np.ceil(self.num_train / self.num_classes))
                    gen_indices = []
                    for l in self.label_values:
                        label_inds = np.where(np.equal(self.input_labels[split], l))[0]
                        rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                        gen_indices += [rand_inds]
                    gen_indices = np.random.permutation(np.hstack(gen_indices))
                else:
                    gen_indices = np.random.permutation(self.num_train)

            elif split == 'valid':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            elif split == 'test':

                # Get indices with the minimum potential
                val_num = min(self.num_test, config.validation_size * config.batch_num)
                if val_num < self.potentials[split].shape[0]:
                    gen_indices = np.argpartition(self.potentials[split], val_num)[:val_num]
                else:
                    gen_indices = np.random.permutation(val_num)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            for p_i in gen_indices:

                # Collect ids
                input_category = self.ids[split][p_i]

                # In case batch is full, yield it and reset it
                if batch_n >= self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tpp_list, axis=0),
                           np.concatenate(tcp_list, axis=0),
                           np.array(tid_list, dtype=np.object),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tpp_list]),
                           np.array([tc.shape[0] for tc in tcp_list]))
                    tpp_list = []
                    tcp_list = []
                    tid_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tpp_list += [self.partial_points[split][p_i].astype(np.float32)]
                tcp_list += [self.complete_points[split][p_i].astype(np.float32)]
                tid_list += [input_category[0]]
                ti_list += [p_i]

                # Update batch size
                batch_n += 1

            yield (np.concatenate(tpp_list, axis=0),
                   np.concatenate(tcp_list, axis=0),
                   np.array(tid_list, dtype=np.object),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tpp_list]),
                   np.array([tc.shape[0] for tc in tcp_list]))

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        gen_types = (tf.float32, tf.float32, tf.string, tf.int32, tf.int32, tf.int32)

        if config.per_cloud_batch:
            used_gen = static_batch_cloud_based_gen
            gen_shapes = (
                [None, 3], [None, 3], [None], [None], [None], [None])
        else:
            used_gen = dynamic_batch_point_based_gen
            gen_shapes = (
                [None, 3], [None, 3], [None], [None], [None], [None])

        return used_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map(stacked_partial, stacked_complete, ids, obj_inds, stacked_partial_lengths,
                   stacked_complete_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_partial: Tensor with size [None, 3] where None is the total number of points
            :param stacked_complete: Tensor with size [None, 3] where None is the total number of points
            :param categories: Tensor with size [None] where None is the number of batch
            :param obj_inds: Tensor with size [None] where None is the number of batch
            :param stacked_partial_lengths: Tensor with size [None] where None is the number of batch
            :param stacked_complete_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch index for each point: [3, 2, 5] --> [0, 0, 0, 1, 1, 2, 2, 2, 2, 2] (but with larger sizes...)
            batch_inds = self.tf_get_batch_inds(stacked_partial_lengths)

            # Augment input points
            # TODO: SHOULD I AUGMENT THE DATA?
            stacked_points, scales, rots = self.tf_augment_input(stacked_partial,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            # if config.per_cloud_batch:
            #     stacked_features = tf.ones((tf.shape(stacked_points)[0], config.num_input_points, 1), dtype=tf.float32)
            # else:
            #     stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                # stacked_features = tf.concat((stacked_features, stacked_points),
                #                              axis=2 if config.per_cloud_batch else 1)
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, stacked_complete), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_completion_inputs(config,
                                                   stacked_points,
                                                   stacked_features,
                                                   stacked_complete,
                                                   stacked_partial_lengths,
                                                   batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds, stacked_partial_lengths, stacked_complete_lengths, ids]

            return input_list

        return tf_map

    def check_input_pipeline_timing(self, config, model):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        n_b = config.batch_num
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = [self.flat_inputs, model.coarse, model.bottleneck_features]

                # Get next inputs
                np_flat_inputs, coarse, output_features = self.sess.run(ops, {model.dropout_prob: 0.5})
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]
                n_b = 0.99 * n_b + 0.01 * batches.shape[0]
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d} => b = {:.1f}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1],
                                         n_b))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_neighbors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        hist_n = 500
        neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                for neighb_mat in neighbors:
                    print(neighb_mat.shape)

                counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                hists = [np.bincount(c, minlength=hist_n) for c in counts]

                neighb_hists += np.vstack(hists)

                print('***********************')
                dispstr = ''
                fmt_l = len(str(int(np.max(neighb_hists)))) + 1
                for neighb_hist in neighb_hists:
                    for v in neighb_hist:
                        dispstr += '{num:{fill}{width}}'.format(num=v, fill=' ', width=fmt_l)
                    dispstr += '\n'
                print(dispstr)
                print('***********************')

                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return


def plot_pcds(filename, pcds, titles, use_color=[], color=None, suptitle='', sizes=None, cmap='Reds', zdir='y',
              xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            clr = color[j]
            if color is None or not use_color[j]:
                clr = pcd[:, 0]

            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=clr, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

# plot_pcds(None, [data[2], data[0]], ['partial', 'gt'], use_color=[0, 0],
#           color=[None, None])
