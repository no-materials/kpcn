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

import tensorflow.contrib.graph_editor as ge

# PLY reader
from utils.ply import write_ply, read_ply

# Configuration class
from utils.config import Config

# CONFIG THESE ###################################
drive_dir = '/content/drive/My Drive/kpcn/'
# drive_results = join(drive_dir, 'results')
drive_results = 'results'


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
        num_votes = 1

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
                        complete_points_op = model.inputs['complete_points']
                    else:
                        raise ValueError('Unsupported dataset')

                    # Run one step of the model
                    t = [time.time()]
                    ops = (all_ops[-1].outputs[0],  # last leaky relu op
                           features_tensor,  # selected leaky relu op
                           complete_points_op,
                           model.inputs['points'],
                           model.inputs['pools'],
                           model.inputs['in_batches'])  # (B x partial_point_count+1) idxs of points for each batch
                    _, stacked_features, complete_points, partial_points, all_pools, in_batches = self.sess.run(ops, {
                        model.dropout_prob: 1.0})
                    t += [time.time()]
                    count += in_batches.shape[0]

                    # Stack all batches
                    max_ind = np.max(in_batches)
                    stacked_batches = []
                    for b_i, b in enumerate(in_batches):
                        stacked_batches += [b[b < max_ind - 0.5] * 0 + b_i]
                    stacked_batches = np.hstack(stacked_batches)

                    # Find batches at wanted layer
                    for l in range(model.config.num_layers - 1):
                        if l >= layer_idx:
                            break
                        stacked_batches = stacked_batches[all_pools[l][:, 0]]

                    # Get each example and update top_activations
                    for b_i, b in enumerate(in_batches):
                        b = b[b < max_ind - 0.5]
                        in_points = partial_points[0][b]
                        features = stacked_features[stacked_batches == b_i]
                        points = partial_points[layer_idx][stacked_batches == b_i]
                        if model.config.dataset in ['ShapeNetPart_multi', 'ModelNet40_classif']:
                            l = partial_points[b_i]

                        self.update_top_activations(features, complete_points[b_i], points, in_points, radius)

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        if model.config.dataset.startswith('S3DIS'):
                            completed = count / (model.config.validation_size * model.config.batch_num)
                        else:
                            completed = count / dataset.num_test
                        message = 'Vote {:d} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(v,
                                             100 * completed,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))
                        # class_names = np.array([dataset.label_to_names[i] for i in range(dataset.num_classes)])
                        # print(class_names[self.top_classes[:, :20]].T)


                except tf.errors.OutOfRangeError:
                    break

        return relu_idx

    def update_top_activations(self, features, label, l_points, input_points, radius, max_computed=60):

        top_num = self.top_features.shape[0]

        # Compute top indice for each feature
        max_indices = np.argmax(features, axis=0)

        # get top_point neighborhoods
        for features_i, idx in enumerate(max_indices[:max_computed]):
            if features[idx, features_i] <= self.top_features[-1, features_i]:
                continue
            # if label in self.top_classes[:, features_i]:
            #     ind0 = np.where(self.top_classes[:, features_i] == label)[0][0]
            #     if features[idx, features_i] <= self.top_features[ind0, features_i]:
            #         continue
            #     elif ind0 < top_num - 1:
            #         self.top_features[ind0:-1, features_i] = self.top_features[ind0 + 1:, features_i]
            #         self.top_classes[ind0:-1, features_i] = self.top_classes[ind0 + 1:, features_i]
            #         for next_i in range(ind0 + 1, top_num):
            #             old_f = join(self.visu_path, self.fmt_str.format(features_i, next_i + 1))
            #             new_f = join(self.visu_path, self.fmt_str.format(features_i, next_i))
            #             if exists(old_f):
            #                 if exists(new_f):
            #                     remove(new_f)
            #                 rename(old_f, new_f)

            # Find indice where new top should be placed
            top_i = np.where(features[idx, features_i] > self.top_features[:, features_i])[0][0]

            # # Update top features
            # if top_i < top_num - 1:
            #     self.top_features[top_i + 1:, features_i] = self.top_features[top_i:-1, features_i]
            #     self.top_features[top_i, features_i] = features[idx, features_i]
            #     self.top_classes[top_i + 1:, features_i] = self.top_classes[top_i:-1, features_i]
            #     self.top_classes[top_i, features_i] = label

            # Find in which batch lays the point
            if self.saving:

                # Get inputs
                l_features = features[:, features_i]
                point = l_points[idx, :]
                dist = np.linalg.norm(input_points - point, axis=1)
                influence = (radius - dist) / radius

                # Project response on input cloud
                if l_points.shape[0] == input_points.shape[0]:
                    responses = l_features
                else:
                    tree = KDTree(l_points, leaf_size=50)
                    nn_k = min(l_points.shape[0], 10)
                    interp_dists, interp_inds = tree.query(input_points, nn_k, return_distance=True)
                    tukeys = np.square(1 - np.square(interp_dists / radius))
                    tukeys[interp_dists > radius] = 0
                    responses = np.sum(l_features[interp_inds] * tukeys, axis=1)

                # Handle last examples
                for next_i in range(top_num - 1, top_i, -1):
                    old_f = join(self.visu_path, self.fmt_str.format(features_i, next_i))
                    new_f = join(self.visu_path, self.fmt_str.format(features_i, next_i + 1))
                    if exists(old_f):
                        if exists(new_f):
                            remove(new_f)
                        rename(old_f, new_f)

                # Save
                filename = join(self.visu_path, self.fmt_str.format(features_i, top_i + 1))
                write_ply(filename,
                          [input_points, influence, responses],
                          ['x', 'y', 'z', 'influence', 'responses'])

    @staticmethod
    def show_activation(path, relu_idx=0, save_video=False):
        """
        This function show the saved input point clouds maximizing the activations. You can also directly load the files
        in a visualization software like CloudCompare.
        In the case of relu_idx = 0 and if gaussian mode, the associated filter is also shown. This function can only
        show the filters for the last saved epoch.
        """

        ################
        # Find the files
        ################

        # Check visu folder
        visu_path = join(drive_results,
                         'visu',
                         'visu_' + path.split('/')[-1],
                         'top_activations',
                         'Relu{:02d}'.format(relu_idx))
        if not exists(visu_path):
            message = 'Relu {:d} activations of the model {:s} not found.'
            raise ValueError(message.format(relu_idx, path.split('/')[-1]))

        # Get the list of files
        feature_files = np.sort([f for f in listdir(visu_path) if f.endswith('.ply')])
        if len(feature_files) == 0:
            message = 'Relu {:d} activations of the model {:s} not found.'
            raise ValueError(message.format(relu_idx, path.split('/')[-1]))

        # Load mode
        config = Config()
        config.load(path)
        mode = config.convolution_mode

        #################
        # Get activations
        #################

        all_points = []
        all_responses = []

        for file in feature_files:
            # Load points
            data = read_ply(join(visu_path, file))
            all_points += [np.vstack((data['x'], data['y'], data['z'])).T]
            all_responses += [data['responses']]

        ###########################
        # Interactive visualization
        ###########################

        # Create figure for features
        fig1 = mlab.figure('Features', bgcolor=(0.5, 0.5, 0.5), size=(640, 480))
        fig1.scene.parallel_projection = False

        # Indices
        global file_i
        file_i = 0

        def update_scene():

            #  clear figure
            mlab.clf(fig1)

            # Plot new data feature
            points = all_points[file_i]
            responses = all_responses[file_i]
            min_response, max_response = np.min(responses), np.max(responses)
            responses = (responses - min_response) / (max_response - min_response)

            # Rescale points for visu
            points = (points * 1.5 / config.in_radius + np.array([1.0, 1.0, 1.0])) * 50.0

            # Show point clouds colorized with activations
            activations = mlab.points3d(points[:, 0],
                                        points[:, 1],
                                        points[:, 2],
                                        responses,
                                        scale_factor=3.0,
                                        scale_mode='none',
                                        vmin=0.1,
                                        vmax=0.9,
                                        figure=fig1)

            # New title
            mlab.title(feature_files[file_i], color=(0, 0, 0), size=0.3, height=0.01)
            text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
            mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
            mlab.orientation_axes()

            return

        def keyboard_callback(vtk_obj, event):
            global file_i

            if vtk_obj.GetKeyCode() in ['g', 'G']:

                file_i = (file_i - 1) % len(all_responses)
                update_scene()

            elif vtk_obj.GetKeyCode() in ['h', 'H']:

                file_i = (file_i + 1) % len(all_responses)
                update_scene()

            return

        # Draw a first plot
        update_scene()
        fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
        mlab.show()

        return
