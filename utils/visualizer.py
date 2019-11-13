# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the visualization
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KDTree
from os import makedirs, remove, rename, listdir
from os.path import exists, join
import time
from mayavi import mlab
import sys

import tensorflow.contrib.graph_editor as ge

# PLY reader
from utils.ply import write_ply, read_ply

# Configuration class
from utils.config import Config

# CONFIG THESE ###################################
drive_dir = '/content/drive/My Drive/kpcn/'
drive_results = join(drive_dir, 'results')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelVisualizer:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Add a softmax operation for predictions
        # model.prob_logits = tf.nn.softmax(model.logits)

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored.")

    # Main visualization methods
    # ------------------------------------------------------------------------------------------------------------------

    def top_relu_activations(self, model, dataset, relu_idx=0, top_num=5):
        """
        Test the model on test dataset to see which points activate the most each neurons in a relu layer
        :param model: model used at training
        :param dataset: dataset used at training
        :param relu_idx: which features are to be visualized
        :param top_num: how many top candidates are kept per features
        """

        #####################################
        # First choose the visualized feature
        #####################################

        # List all relu ops
        all_ops = [op for op in tf.get_default_graph().get_operations() if op.name.startswith('KernelPointNetwork')
                   and op.name.endswith('LeakyRelu')]

        #  List all possible Relu indices
        print('\nPossible Relu indices:')
        for i, t in enumerate(all_ops):
            print(i, ': ', t.name)

        # Print the chosen one
        if relu_idx is not None:
            features_tensor = all_ops[relu_idx].outputs[0]
        else:
            relu_idx = int(input('Choose a Relu index: '))
            features_tensor = all_ops[relu_idx].outputs[0]

        # Get parameters
        layer_idx = int(features_tensor.name.split('/')[1][6:])
        if 'strided' in all_ops[relu_idx].name and not ('strided' in all_ops[relu_idx + 1].name):
            layer_idx += 1
        features_dim = int(features_tensor.shape[1])
        radius = model.config.first_subsampling_dl * model.config.density_parameter * (2 ** layer_idx)

        print('You chose to compute the output of operation named:\n' + all_ops[relu_idx].name)
        print('\nIt contains {:d} features.'.format(int(features_tensor.shape[1])))

        print('\n****************************************************************************')

        #####################
        # Initiate containers
        #####################

        # Initiate containers
        self.top_features = -np.ones((top_num, features_dim))
        self.top_classes = -np.ones((top_num, features_dim), dtype=np.int32)
        self.saving = model.config.saving

        # Testing parameters
        num_votes = 3

        # Create visu folder
        self.visu_path = None
        self.fmt_str = None
        if model.config.saving:
            self.visu_path = join(drive_results,
                                  'visu',
                                  'visu_' + model.saving_path.split('/')[-1],
                                  'top_activations',
                                  'Relu{:02d}'.format(relu_idx))
            self.fmt_str = 'f{:04d}_top{:02d}.ply'
            if not exists(self.visu_path):
                makedirs(self.visu_path)

        # *******************
        # Network predictions
        # *******************

        mean_dt = np.zeros(2)
        last_display = time.time()
        for v in range(num_votes):

            # Run model on all test examples
            # ******************************

            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)
            count = 0

            while True:
                try:

                    if model.config.dataset.startswith('ShapeNetV1'):
                        label_op = model.inputs['complete_points']
                    else:
                        raise ValueError('Unsupported dataset')

                    # Run one step of the model
                    t = [time.time()]
                    ops = (all_ops[-1].outputs[0],  # last leaky relu op
                           features_tensor,  # selected leaky relu op
                           label_op,
                           model.inputs['points'],
                           model.inputs['pools'],
                           model.inputs['in_batches'])
                    _, stacked_features, labels, all_points, all_pools, in_batches = self.sess.run(ops, {
                        model.dropout_prob: 1.0})
                    t += [time.time()]
                    count += in_batches.shape[0]

                    # TODO: WIP


                except tf.errors.OutOfRangeError:
                    break

        return relu_idx
