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

    def __init__(self, class_name, input_threads=8):
        """
        Initiation method. Give the name of the object class to complete (for example 'Airplane') or 'multi' to complete
        all objects with a single model.
        """
        Dataset.__init__(self, 'ShapeNetV1_' + class_name)

        ###########################
        # Object classes parameters
        ###########################

        self.synset_to_category = {
            '02691156': 'Airplane',
            '02773838': 'Bag',
            '02954340': 'Cap',
            '02958343': 'Car',
            '03001627': 'Chair',
            '03261776': 'Earphone',
            '03467517': 'Guitar',
            '03624134': 'Knife',
            '03636649': 'Lamp',
            '03642806': 'Laptop',
            '03790512': 'Motorbike',
            '03797390': 'Mug',
            '03948459': 'Pistol',
            '04099429': 'Rocket',
            '04225987': 'Skateboard',
            '04379243': 'Table',
            '02933112': 'Cabinet',
            '04256520': 'Sofa',
            '04530566': 'Boat',
            '02818832': 'Bed',
            '02828884': 'Bench',
            '02871439': 'Bookshelf',
            '02924116': 'Bus',
            '03211117': 'Display',
            '04004475': 'Printer',
            '04401088': 'Telephone'
        }

        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Number of parts for each object
        # self.num_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

        # Type of dataset (one of the class names or 'multi')
        self.ShapeNetV1Type = class_name

        if self.ShapeNetV1Type == 'multi':

            # Number of models
            self.network_model = 'multi_completion'
            self.num_train = 28974
            self.num_test = 1200

        elif self.ShapeNetV1Type in self.category_names:

            # Number of models computed when init_subsample_clouds is called
            self.network_model = 'completion'
            self.num_train = None
            self.num_test = None

        else:
            raise ValueError('Unsupported ShapenetV1 object class : \'{:s}\''.format(self.ShapeNetV1Type))

        # Partial, complete point clouds & categories used for each split
        self.partial_points = {}
        self.complete_points = {}
        self.categories = {}

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

        # TODO: MY LABELS ARE THE COMPLETE GT POINT CLOUDS...
        # Initiate containers
        self.partial_points = {'training': [], 'validation': [], 'test': []}
        self.complete_points = {'training': [], 'validation': [], 'test': []}
        self.categories = {'training': [], 'validation': [], 'test': []}

        ################
        # Training files
        ################

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading training points')
        filename = join(self.data_path, 'train_{:.3f}_record.pkl'.format(subsampling_parameter))

        if exists(filename):
            with open(filename, 'rb') as file:
                self.partial_points['training'], \
                self.complete_points['training'], \
                self.categories['training'] = pickle.load(file)

        # Else compute them from original points
        else:
            # Collect complete & partial training file data
            split_type = 'train'

            with open(self.train_split_file) as file:
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
                    self.complete_points['training'] += [sub_complete_points]
                else:
                    self.complete_points['training'] += [complete_points]

                # For each scan, read partial ply data, if subsample param exists, save subsampled partial pc
                for s in range(self.num_scans):
                    partial_data = read_ply(
                        join(self.data_path, split_type, 'partial', cat_id, model_id, "%s.ply" % s))
                    partial_points = np.vstack((partial_data['x'], partial_data['y'], partial_data['z'])).astype(
                        np.float32).T

                    if subsampling_parameter > 0:
                        sub_partial_points = grid_subsampling(partial_points, sampleDl=subsampling_parameter)
                        self.partial_points['training'] += [sub_partial_points]
                    else:
                        self.partial_points['training'] += [partial_points]

            # Get training classes used
            categories = [cat_dir_name for cat_dir_name in listdir(join(self.data_path, split_type, 'complete'))
                          if not cat_dir_name.startswith('.')]
            self.categories['training'] = np.array([self.synset_to_category[cat] for cat in categories])

            # Collect complete & partial validation file data
            split_type = 'valid'

            with open(self.valid_split_file) as file:
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
                    self.complete_points['training'] += [sub_complete_points]
                else:
                    self.complete_points['training'] += [complete_points]

                # For each scan, read partial ply data, if subsample param exists, save subsampled partial pc
                for s in range(self.num_scans):
                    partial_data = read_ply(
                        join(self.data_path, split_type, 'partial', cat_id, model_id, "%s.ply" % s))
                    partial_points = np.vstack((partial_data['x'], partial_data['y'], partial_data['z'])).astype(
                        np.float32).T

                    if subsampling_parameter > 0:
                        sub_partial_points = grid_subsampling(partial_points, sampleDl=subsampling_parameter)
                        self.partial_points['training'] += [sub_partial_points]
                    else:
                        self.partial_points['training'] += [partial_points]

            # Get training classes used
            categories = [cat_dir_name for cat_dir_name in listdir(join(self.data_path, split_type, 'complete'))
                          if not cat_dir_name.startswith('.')]
            self.categories['training'] = np.hstack((self.categories['training'],
                                                     np.array([self.synset_to_category[cat] for cat in categories])))

            # Save training pickle for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.partial_points['training'],
                             self.complete_points['training'],
                             self.categories['training']), file)

        lengths = [p.shape[0] for p in self.partial_points['training']]
        lengths.extend([p.shape[0] for p in self.complete_points['training']])
        sizes = [l * 4 * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        ############
        # Test files
        ############

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        print('\nLoading test points')
        filename = join(self.data_path, 'test_{:.3f}_record.pkl'.format(subsampling_parameter))
        if exists(filename):
            with open(filename, 'rb') as file:
                self.partial_points['test'], \
                self.complete_points['test'], \
                self.categories['test'] = pickle.load(file)

        # Else compute them from original points
        else:
            # Collect complete & partial test file data
            split_type = 'test'

            with open(self.test_split_file) as file:
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
                    self.complete_points['test'] += [sub_complete_points]
                else:
                    self.complete_points['test'] += [complete_points]

                # For each scan, read partial ply data, if subsample param exists, save subsampled partial pc
                for s in range(self.num_scans):
                    partial_data = read_ply(
                        join(self.data_path, split_type, 'partial', cat_id, model_id, "%s.ply" % s))
                    partial_points = np.vstack((partial_data['x'], partial_data['y'], partial_data['z'])).astype(
                        np.float32).T

                    if subsampling_parameter > 0:
                        sub_partial_points = grid_subsampling(partial_points, sampleDl=subsampling_parameter)
                        self.partial_points['test'] += [sub_partial_points]
                    else:
                        self.partial_points['test'] += [partial_points]

            # Get test classes used
            categories = [cat_dir_name for cat_dir_name in listdir(join(self.data_path, split_type, 'complete'))
                          if not cat_dir_name.startswith('.')]
            self.categories['test'] = np.array([self.synset_to_category[cat] for cat in categories])

            # Save test pickle for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.partial_points['test'],
                             self.complete_points['test'],
                             self.categories['test']), file)

        lengths = [p.shape[0] for p in self.partial_points['test']]
        lengths.extend([p.shape[0] for p in self.complete_points['test']])
        sizes = [l * 4 * 3 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        #######################################
        # Eliminate unconsidered object classes
        #######################################

        # Eliminate unconsidered classes if not multi
        if self.ShapeNetV1Type in self.category_names:
            # Index of wanted category
            wanted_synset = self.category_to_synset[self.ShapeNetV1Type]

            # Manage training points
            boolean_mask = self.categories['training'] == self.ShapeNetV1Type
            self.categories['training'] = self.categories['training'][boolean_mask]
            self.partial_points['training'] = np.array(self.partial_points['training'])[boolean_mask]
            self.complete_points['training'] = np.array(self.complete_points['training'])[boolean_mask]
            self.num_train = len(self.categories['training'])

            # Manage test points
            boolean_mask = self.categories['test'] == self.ShapeNetV1Type
            self.categories['test'] = self.categories['test'][boolean_mask]
            self.partial_points['test'] = np.array(self.partial_points['test'])[boolean_mask]
            self.complete_points['test'] = np.array(self.complete_points['test'])[boolean_mask]
            self.num_train = len(self.categories['test'])

        # Test = validation
        # TODO: WHY?
        self.categories['validation'] = self.categories['test']
        self.partial_points['validation'] = self.partial_points['test']
        self.complete_points['validation'] = self.complete_points['test']

        return


if __name__ == '__main__':
    ds = ShapeNetV1Dataset('multi')

    # Create subsampled input clouds
    dl0 = 0.02
    ds.load_subsampled_clouds(dl0)
