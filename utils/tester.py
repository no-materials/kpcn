# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
from os import makedirs
from os.path import exists, join
import time

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import chamfer, earth_mover

# Vis
from utils.visualizer import plot_pc_compare_views


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        # TODO: add auto check device
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
            print("Model restored from " + restore_snap)

        # define distance ops
        gt_ds = tf.reshape(model.inputs['complete_points'], [-1, model.config.num_gt_points, 3])
        gt_ds_trunc = gt_ds[:, :model.config.num_coarse, :]

        self.coarse_earth_mover = earth_mover(model.coarse, gt_ds_trunc)
        self.fine_chamfer = chamfer(model.fine, gt_ds)

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_completion(self, model, dataset, num_votes=100):
        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Number of classes predicted by the model
        nc_model = model.config.num_classes

        # Initiate votes
        average_probs = np.zeros((len(dataset.input_labels['test']), nc_model))
        average_counts = np.zeros((len(dataset.input_labels['test']), nc_model))

        mean_dt = np.zeros(2)
        last_display = time.time()

        # Run model on all test examples
        # ******************************

        # Initiate result containers
        coarse_em_list = []
        fine_cd_list = []
        coarse_list = []
        fine_list = []
        complete_points_list = []
        partial_points_list = []
        obj_inds = []

        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.coarse_earth_mover, self.fine_chamfer, model.coarse, model.fine, model.complete_points,
                       model.inputs['points'], model.inputs['object_inds'])
                coarse_em, fine_cd, coarse, fine, complete, partial, inds = self.sess.run(ops, {
                    model.dropout_prob: 1.0})
                t += [time.time()]

                # Get distances and obj_indexes
                coarse_em_list += [coarse_em]
                fine_cd_list += [fine_cd]
                coarse_list += [coarse]
                fine_list += [fine]
                complete_points_list += [complete]
                partial_points_list += [partial]
                obj_inds += [inds]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Test : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * len(obj_inds) / dataset.num_test,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        coarse_em_mean = np.mean(coarse_em_list)
        fine_cd_mean = np.mean(fine_cd_list)
        print('Test distances\nMean (Fine) Chamfer: {:4.3f}\tMean (Coarse) Earth Mover: {:4.3f}'.format(
            fine_cd_mean,
            coarse_em_mean))

        if model.config.saving:
            if not exists(join(model.saving_path, 'visu')):
                makedirs(join(model.saving_path, 'visu'))

            all_pcs = [partial_points_list, coarse_list, fine_list, complete_points_list]
            visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']
            for i in range(0, len(coarse_list), 20):
                plot_path = join(model.saving_path, 'visu', 'test', '%d.png' % i)  # TODO: add ids as plot filename
                pcs = [x[i] for x in all_pcs]
                partial_temp = pcs[0][0][:model.config.num_input_points, :]
                coarse_temp = pcs[1][0, :, :]
                fine_temp = pcs[2][0, :, :]
                complete_temp = pcs[3][:model.config.num_gt_points, :]
                final_pcs = [partial_temp, coarse_temp, fine_temp, complete_temp]
                plot_pc_compare_views(plot_path, final_pcs, visualize_titles)

        return
