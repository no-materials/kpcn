import tensorflow as tf
import numpy as np
import time
import json
import pickle
from sklearn.neighbors import KDTree
import open3d as o3d

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


class KittiDataset(Dataset):
    """
    Kitti dataset for completion task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, batch_num, input_pts, dataset_path, pickle_path, input_threads=8):
        """
        Initiation method.
        """
        Dataset.__init__(self, 'kitti')

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

        # Partial, ids used for each car
        self.partial_points = {}
        self.ids = {}

        ##########################
        # Parameters for the files
        ##########################

        # Path of the dataset src folder
        self.dataset_path = dataset_path
        self.pickle_path = pickle_path

        self.batch_num = batch_num

        # Number of threads
        self.num_threads = input_threads

        self.input_pts = input_pts

        self.pcd_dir = join(self.dataset_path, 'cars')
        self.bbox_dir = join(self.dataset_path, 'bboxes')
        self.tracklets_dir = join(self.dataset_path, 'tracklets')

        self.num_cars = 2401  # TODO: fix hardcoded value

    def load_cloud(self, fname):
        pcd = o3d.io.read_point_cloud(join(self.pcd_dir, fname))
        partial_cloud = np.array(pcd.points)
        cloud_meta = '{}'.format(fname.split('/')[-1])
        return partial_cloud, cloud_meta,

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        self.partial_points = {'test': []}
        self.ids = {'test': []}

        split_type = 'test'

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading %s points' % split_type)
        filename = join(self.pickle_path, '{0:s}_{1:.3f}_record.pkl'.format('test_kitti', subsampling_parameter))

        if exists(filename):
            with open(filename, 'rb') as file:
                self.partial_points[split_type], \
                self.ids[split_type] = pickle.load(file)

        # Else compute them from original points
        else:
            print('Recomputing test_kitti pkl file')
            for file_iter, file_path in enumerate([f for f in listdir(self.pcd_dir) if f.endswith('.pcd')]):
                print('Car {}/{}'.format(file_iter, self.num_cars))
                # Call loading functions
                data = self.load_cloud(file_path)
                bbox = np.loadtxt(join(self.bbox_dir, '%s.txt' % file_path.split('.')[0]))

                # Normalize clouds, calculate center, rotation and scale
                center = (bbox.min(0) + bbox.max(0)) / 2
                bbox -= center
                yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
                rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                     [np.sin(yaw), np.cos(yaw), 0],
                                     [0, 0, 1]])
                bbox = np.dot(bbox, rotation)
                scale = bbox[3, 0] - bbox[0, 0]
                bbox /= scale

                partial = np.dot(data[0] - center, rotation) / scale
                partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

                if subsampling_parameter > 0:
                    sub_partial_points = grid_subsampling(partial.astype(np.float32),
                                                          sampleDl=subsampling_parameter)
                    padded_sub_partial = pad_cloudN(sub_partial_points, self.input_pts)
                    self.partial_points[split_type] += [padded_sub_partial]
                    self.ids[split_type] += [data[1]]
                    # plot_pcds(None, [partial], ['partial'], use_color=[0], color=[None])
                else:
                    padded_partial = pad_cloudN(partial, self.input_pts)
                    self.partial_points[split_type] += [padded_partial]
                    self.ids[split_type] += [data[1]]
                    # plot_pcds(None, [partial, padded_partial], ['partial', 'padded'], use_color=[0, 0], color=[None, None])

            # Save split pickle for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.partial_points[split_type],
                             self.ids[split_type]), file)

            lengths = [p.shape[0] for p in self.partial_points[split_type]]
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
        self.potentials[split] = np.random.rand(len(self.ids[split])) * 1e-3

        ################
        # Def generators
        ################
        def dynamic_batch_point_based_gen():

            # Initiate concatenation lists
            tpp_list = []  # partial points
            tid_list = []  # ids
            ti_list = []  # cloud index
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'test':

                # Get indices with the minimum potential
                gen_indices = np.random.permutation(self.num_cars)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            for p_i in gen_indices:

                # Get points
                new_partial_points = self.partial_points[split][p_i].astype(np.float32)
                n = new_partial_points.shape[0]  # num of points of selected partial point cloud

                # Collect labels
                input_category = self.ids[split][p_i]

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tpp_list, axis=0),
                           np.array(tid_list),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tpp_list]))
                    tpp_list = []
                    tid_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tpp_list += [new_partial_points]
                tid_list += [input_category[0]]
                ti_list += [p_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tpp_list, axis=0),
                   np.array(tid_list),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tpp_list]))

        def static_batch_cloud_based_gen():

            # Initiate concatenation lists
            tpp_list = []  # partial points
            tid_list = []  # categories
            ti_list = []  # cloud index
            batch_n = 0

            # Initiate parameters depending on the chosen split
            if split == 'test':

                # Get indices with the minimum potential
                gen_indices = np.random.permutation(self.num_cars)

                # Update potentials
                self.potentials[split][gen_indices] += 1.0

            else:
                raise ValueError('Wrong split argument in data generator: ' + split)

            # Generator loop
            for p_i in gen_indices:

                # Collect ids
                input_id = self.ids[split][p_i]

                # In case batch is full, yield it and reset it
                if batch_n >= self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tpp_list, axis=0),
                           np.array(tid_list, dtype=np.object),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tpp_list]))
                    tpp_list = []
                    tid_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tpp_list += [self.partial_points[split][p_i].astype(np.float32)]
                tid_list += [input_id]
                ti_list += [p_i]

                # Update batch size
                batch_n += 1

            yield (np.concatenate(tpp_list, axis=0),
                   np.array(tid_list, dtype=np.object),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tpp_list]))

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        gen_types = (tf.float32, tf.string, tf.int32, tf.int32)

        if config.per_cloud_batch:
            used_gen = static_batch_cloud_based_gen
            gen_shapes = (
                [None, 3], [None], [None], [None])
        else:
            used_gen = dynamic_batch_point_based_gen
            gen_shapes = (
                [None, 3], [None], [None], [None])

        return used_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map(stacked_partial, ids, obj_inds, stacked_partial_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_partial: Tensor with size [None, 3] where None is the total number of points
            :param ids: Tensor with size [None] where None is the number of batch
            :param obj_inds: Tensor with size [None] where None is the number of batch
            :param stacked_partial_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch index for each point: [3, 2, 5] --> [0, 0, 0, 1, 1, 2, 2, 2, 2, 2] (but with larger sizes...)
            batch_inds = self.tf_get_batch_inds(stacked_partial_lengths)

            scales = None
            rots = None

            stacked_features = tf.ones((tf.shape(stacked_partial)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_partial), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, and 4 (without and with XYZ)')

            # Get the whole input list
            input_list = self.tf_completion_inputs(config,
                                                   stacked_partial,
                                                   stacked_features,
                                                   None,
                                                   stacked_partial_lengths,
                                                   batch_inds)

            # Add dummy scale and rotation for testing
            input_list += [tf.zeros((0, 1), dtype=tf.int32), tf.zeros((0, 1), dtype=tf.int32), obj_inds,
                           stacked_partial_lengths, tf.zeros((0, 1), dtype=tf.int32), ids]

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
    fig = plt.figure(figsize=(len(pcds) * 3, 1))
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
