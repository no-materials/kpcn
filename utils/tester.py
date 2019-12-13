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
from os.path import exists, join, dirname
import time

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import chamfer, earth_mover

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    def test_completion(self, model, dataset, on_val, num_votes=100):

        # Initialise iterator with data
        if on_val:
            self.sess.run(dataset.val_init_op)
            cardinal = dataset.num_valid
        else:
            self.sess.run(dataset.test_init_op)
            cardinal = dataset.num_test

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
        ids_list = []

        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.coarse_earth_mover, self.fine_chamfer, model.coarse, model.fine, model.complete_points,
                       model.inputs['points'], model.inputs['object_inds'], model.inputs['ids'])
                coarse_em, fine_cd, coarse, fine, complete, partial, inds, idss = self.sess.run(ops, {
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
                ids_list += [idss]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Test : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * len(obj_inds) / cardinal,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        coarse_em_mean = np.mean(coarse_em_list)
        fine_cd_mean = np.mean(fine_cd_list)
        print('Test distances\nMean (Fine) Chamfer: {:4.5f}\tMean (Coarse) Earth Mover: {:4.5f}'.format(
            fine_cd_mean,
            coarse_em_mean))

        if model.config.saving:
            if not exists(join(model.saving_path, 'visu', 'test')):
                makedirs(join(model.saving_path, 'visu', 'test'))

            all_pcs = [partial_points_list, coarse_list, fine_list, complete_points_list]
            all_dist = [coarse_em_list, fine_cd_list]
            visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']
            for i, id_batch_np in enumerate(ids_list):
                plot_path = join(model.saving_path, 'visu', 'test', '%s.png' % id_batch_np[0].decode().split(".")[0])
                if not exists(dirname(plot_path)):
                    makedirs(dirname(plot_path))
                pcs = [x[i] for x in all_pcs]
                dists = [d[i] for d in all_dist]
                suptitle = 'Coarse EMD = %d / Fine CD = %d' % (dists[0], dists[1])
                partial_temp = pcs[0][0][:model.config.num_input_points, :]
                coarse_temp = pcs[1][0, :, :]
                fine_temp = pcs[2][0, :, :]
                complete_temp = pcs[3][:model.config.num_gt_points, :]
                final_pcs = [partial_temp, coarse_temp, fine_temp, complete_temp]
                self.plot_pc_compare_views(plot_path, final_pcs, visualize_titles, suptitle=suptitle)

        return

    def test_kitti_completion(self, model, dataset):

        # Initialise iterator with data
        self.sess.run(dataset.test_init_op)
        cardinal = dataset.num_cars

        mean_dt = np.zeros(2)
        last_display = time.time()

        # Run model on all test examples
        # ******************************

        # Initiate result containers
        coarse_list = []
        fine_list = []
        partial_points_list = []
        obj_inds = []
        ids_list = []

        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (
                    model.coarse, model.fine, model.inputs['points'], model.inputs['object_inds'], model.inputs['ids'])
                coarse, fine, partial, inds, idss = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get results and append to list
                coarse_list += [coarse]
                fine_list += [fine]
                partial_points_list += [partial]
                obj_inds += [inds]
                ids_list += [idss]

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Test : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * len(obj_inds) / cardinal,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        if model.config.saving:
            if not exists(join(model.saving_path, 'visu', 'kitti')):
                makedirs(join(model.saving_path, 'visu', 'kitti'))

            all_pcs = [partial_points_list, coarse_list, fine_list]
            visualize_titles = ['input', 'coarse output', 'fine output']
            for i, id_batch_np in enumerate(ids_list):
                plot_path = join(model.saving_path, 'visu', 'kitti', '%s.png' % id_batch_np[0].decode().split(".")[0])
                if not exists(dirname(plot_path)):
                    makedirs(dirname(plot_path))
                pcs = [x[i] for x in all_pcs]
                partial_temp = pcs[0][0][:model.config.num_input_points, :]
                coarse_temp = pcs[1][0, :, :]
                fine_temp = pcs[2][0, :, :]
                final_pcs = [partial_temp, coarse_temp, fine_temp]
                self.plot_pc_compare_views(plot_path, final_pcs, visualize_titles)

        return

    @staticmethod
    def plot_pc_compare_views(filename, pcs, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                              xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
        if sizes is None:
            sizes = [0.5 for i in range(len(pcs))]
        fig = plt.figure(figsize=(len(pcs) * 3, 9))
        for i in range(3):
            elev = 30
            azim = -45 + 90 * i
            for j, (pc, size) in enumerate(zip(pcs, sizes)):
                color = pc[:, 0]
                ax = fig.add_subplot(3, len(pcs), i * len(pcs) + j + 1, projection='3d')
                ax.view_init(elev, azim)
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
                ax.set_title(titles[j])
                ax.set_axis_off()
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
        plt.suptitle(suptitle)
        fig.savefig(filename)
        plt.close(fig)
