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

from preprocess.preprocess_partial_pc import num_scans


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


class ShapeNetV1Dataset(Dataset):
    """
    ShapeNetV1 dataset for completion task. Can handle both unique object class or multi classes models.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8):
        """
        Initiation method.
        """
        Dataset.__init__(self, 'ShapeNetV1')

        ###########################
        # Object classes parameters
        ###########################

        # self.synset_to_category = {
        #     '02691156': 'Airplane',
        #     '02773838': 'Bag',
        #     '02954340': 'Cap',
        #     '02958343': 'Car',
        #     '03001627': 'Chair',
        #     '03261776': 'Earphone',
        #     '03467517': 'Guitar',
        #     '03624134': 'Knife',
        #     '03636649': 'Lamp',
        #     '03642806': 'Laptop',
        #     '03790512': 'Motorbike',
        #     '03797390': 'Mug',
        #     '03948459': 'Pistol',
        #     '04099429': 'Rocket',
        #     '04225987': 'Skateboard',
        #     '04379243': 'Table',
        #     '02933112': 'Cabinet',
        #     '04256520': 'Sofa',
        #     '04530566': 'Boat',
        #     '02818832': 'Bed',
        #     '02828884': 'Bench',
        #     '02871439': 'Bookshelf',
        #     '02924116': 'Bus',
        #     '03211117': 'Display',
        #     '04004475': 'Printer',
        #     '04401088': 'Telephone'
        # }

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
        self.num_train = 57946  # account for each scan
        self.num_valid = 1600  # account for each scan
        self.num_test = 2400  # account for each scan

        # Partial, complete point clouds & categories used for each split
        self.partial_points = {}
        self.complete_points = {}
        self.categories = {}  # Unused?...

        ##########################
        # Parameters for the files
        ##########################

        # Path of the dataset src folder
        self.dataset_path = '/Volumes/warm_blue/datasets/ShapeNetV1'

        # Path to preprocessed data folder
        self.data_path = join(dirname(dirname(realpath(__file__))),
                              'data',
                              'shapenetV1')

        # Split file paths
        self.train_split_file = join(self.data_path, 'train.list')
        self.valid_split_file = join(self.data_path, 'valid.list')
        self.test_split_file = join(self.data_path, 'test.list')
        self.test_novel_split_file = join(self.data_path, 'test_novel.list')
        self.one_model_split_file = join(self.data_path, 'one_model.list')  # FOR DEBUG

        # Number of threads
        self.num_threads = input_threads

        # Number of scans from virtual depth rendering during partial pc generation in preprocess step
        self.num_scans = 2
        assert self.num_scans == num_scans

        # TODO: I should probably center & rescale to 1m the plys before saving them to disk (I should test via blender)

        return

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # MY LABELS ARE THE COMPLETE GT POINT CLOUDS
        # Initiate containers
        self.partial_points = {'train': [], 'valid': [], 'test': []}
        self.complete_points = {'train': [], 'valid': [], 'test': []}
        self.categories = {'train': [], 'valid': [], 'test': []}

        for split_type in ['train', 'valid', 'test']:

            # Restart timer
            t0 = time.time()

            # Load wanted points if possible
            print('\nLoading %s points' % split_type)
            filename = join(self.data_path, '{0:s}_{1:.3f}_record.pkl'.format(split_type, subsampling_parameter))

            if exists(filename):
                with open(filename, 'rb') as file:
                    self.partial_points[split_type], \
                    self.complete_points[split_type], \
                    self.categories[split_type] = pickle.load(file)

            # Else compute them from original points
            else:
                # Collect complete & partial file data
                with open(join(self.data_path, '%s.list' % split_type)) as file:
                    model_list = file.read().splitlines()
                    file.close()

                for i, cat_model_id in enumerate(model_list):
                    cat_id, model_id = cat_model_id.split('/')

                    # Read complete ply data, if subsample param exists, save subsampled complete pc, else save original
                    complete_data = read_ply(join(self.data_path, split_type, 'complete', cat_id, "%s.ply" % model_id))
                    complete_points = np.vstack((complete_data['x'], complete_data['y'], complete_data['z'])).astype(
                        np.float32).T

                    if subsampling_parameter > 0:
                        sub_complete_points = grid_subsampling(complete_points, sampleDl=subsampling_parameter)

                    # For each scan, read partial ply data, if subsample param exists, save subsampled partial pc
                    for s in range(self.num_scans):
                        partial_data = read_ply(
                            join(self.data_path, split_type, 'partial', cat_id, model_id, "%s.ply" % s))
                        partial_points = np.vstack((partial_data['x'], partial_data['y'], partial_data['z'])).astype(
                            np.float32).T

                        if subsampling_parameter > 0:
                            sub_partial_points = grid_subsampling(partial_points, sampleDl=subsampling_parameter)
                            self.partial_points[split_type] += [sub_partial_points]
                            # complete points & synsets will be duplicated/matched for each scan
                            self.complete_points[split_type] += [sub_complete_points]
                            self.categories[split_type] += [cat_id]
                        else:
                            self.partial_points[split_type] += [partial_points]
                            self.complete_points[split_type] += [complete_points]
                            self.categories[split_type] += [cat_id]

                # Save split pickle for later use
                with open(filename, 'wb') as file:
                    pickle.dump((self.partial_points[split_type],
                                 self.complete_points[split_type],
                                 self.categories[split_type]), file)

            lengths = [p.shape[0] for p in self.partial_points[split_type]]
            lengths.extend([p.shape[0] for p in self.complete_points[split_type]])
            sizes = [l * 4 * 3 for l in lengths]
            print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        self.num_train = len(self.categories['train'])
        self.num_valid = len(self.categories['valid'])
        self.num_test = len(self.categories['test'])

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
        self.potentials[split] = np.random.rand(len(self.categories[split])) * 1e-3

        ################
        # Def generators
        ################
        def random_balanced_gen():

            # Initiate concatenation lists
            tpp_list = []  # partial points
            tcp_list = []  # complete points
            tcat_list = []  # categories
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
                input_category = self.categories[split][p_i]

                # In case batch is full, yield it and reset it
                # TODO: Currently I'm checking the partial cloud's batch limit, not the partial & complete's sum
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(tpp_list, axis=0),
                           np.concatenate(tcp_list, axis=0),
                           np.array(tcat_list, dtype=np.unicode_),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tpp_list]))
                    tpp_list = []
                    tcp_list = []
                    tcat_list = []
                    ti_list = []
                    batch_n = 0

                # Add data to current batch
                tpp_list += [new_partial_points]
                tcp_list += [new_complete_points]
                tcat_list += [input_category]
                ti_list += [p_i]

                # Update batch size
                batch_n += n

            yield (np.concatenate(tpp_list, axis=0),
                   np.concatenate(tcp_list, axis=0),
                   np.array(tcat_list, dtype=np.unicode_),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tpp_list]))

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        gen_types = (tf.float32, tf.float32, tf.string, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map(stacked_partial, stacked_complete, categories, obj_inds, stacked_partial_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_partial: Tensor with size [None, 3] where None is the total number of points
            :param stacked_complete: Tensor with size [None, 3] where None is the total number of points
            :param categories: Tensor with size [None] where None is the number of batch
            :param obj_inds: Tensor with size [None] where None is the number of batch
            :param stacked_partial_lengths: Tensor with size [None] where None is the number of batch
            """

            # Get batch index for each point: [3, 2, 5] --> [0, 0, 0, 1, 1, 2, 2, 2, 2, 2] (but with larger sizes...)
            batch_inds = self.tf_get_batch_inds(stacked_partial_lengths)

            # Augment input points
            # TODO: SHOULD I AUGMENT THE DATA?
            stacked_points, scales, rots = self.tf_augment_input(stacked_partial,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_points, stacked_complete), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

            # Get the whole input list
            # TODO: Pass complete pc here...classification uses labels, I use complete pcs. Do I RLLY need to pass cats?
            input_list = self.tf_completion_inputs(config,
                                                   stacked_points,
                                                   stacked_features,
                                                   stacked_complete,
                                                   categories,
                                                   stacked_partial_lengths,
                                                   batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]

            return input_list

        return tf_map
