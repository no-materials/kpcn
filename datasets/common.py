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

        # Number of different labels
        self.num_classes = 0

        # The different label values
        self.label_values = []

        # The different label names
        self.label_names = []

        # Dict from labels to [0:num_classes] indices
        self.label_to_idx = {}

        # Dict from labels to names
        self.label_to_names = {}

        # Other Parameters
        # ****************

        # Max number of convolution neighbors
        self.neighborhood_limits = None

        # List of ignored label for this dataset
        self.ignored_labels = []

        # Type of task performed on this dataset
        self.network_model = ''

        # Number of threads used in input pipeline
        self.num_threads = 1


