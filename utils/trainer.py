# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import psutil
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import chamfer, earth_mover

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#

class ModelTrainer:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Create a session for running Ops on the Graph.
        # TODO: add auto check device
        on_CPU = False
        # on_CPU = True
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Add training ops
        self.add_train_ops(model)

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        print('*************************************')
        summ = 0
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='KernelPointNetwork'):
            # print(var.name, var.shape)
            summ += np.prod(var.shape)
        print('total parameters : ', summ)
        print('*************************************')

        print('*************************************')
        summ = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork'):
            # print(var.name, var.shape)
            summ += np.prod(var.shape)
        print('total parameters : ', summ)
        print('*************************************')


        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = os.path.join(model.config.saving_path, 'snapshots/snap-53444')
        if restore_snap is not None:
            exclude_vars = ['softmax', 'head_unary_conv', '/fc/']
            restore_vars = my_vars
            for exclude_var in exclude_vars:
                restore_vars = [v for v in restore_vars if exclude_var not in v.name]
            print(restore_vars)
            restorer = tf.train.Saver(restore_vars)
            restorer.restore(self.sess, restore_snap)
            print("Model restored.")

    def add_train_ops(self, model):
        """
        Add training ops on top of the model
        """

        ##############
        # Training ops
        ##############

        with tf.variable_scope('optimizer'):

            # Learning rate as a Variable so we can modify it
            self.learning_rate = tf.Variable(model.config.learning_rate, trainable=False, name='learning_rate')

            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, model.config.momentum)

            # Training step op
            gvs = optimizer.compute_gradients(model.loss)

            if model.config.grad_clip_norm > 0:

                # Get gradient for deformable convolutions and scale them
                scaled_gvs = []
                for grad, var in gvs:
                    if 'offset_conv' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    if 'offset_mlp' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    else:
                        scaled_gvs.append((grad, var))

                # Clipping each gradient independantly
                capped_gvs = [(tf.clip_by_norm(grad, model.config.grad_clip_norm), var) for grad, var in scaled_gvs]

                # Clipping the whole network gradient (problematic with big network where grad == inf)
                # capped_grads, global_norm = tf.clip_by_global_norm([grad for grad, var in gvs], self.config.grad_clip_norm)
                # vars = [var for grad, var in gvs]
                # capped_gvs = [(grad, var) for grad, var in zip(capped_grads, vars)]

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(capped_gvs)

            else:
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(gvs)

        ############
        # Result ops
        ############

        # Add the Op to compare the distances to the gt during evaluation.
        with tf.variable_scope('results'):

            gt_ds = tf.reshape(model.inputs['complete_points'], [-1, model.config.num_gt_points, 3])
            gt_ds_trunc = gt_ds[:, :model.config.num_coarse, :]

            self.coarse_earth_mover = earth_mover(model.coarse, gt_ds_trunc)
            self.fine_chamfer = chamfer(model.fine, gt_ds)

            self.mixed_loss = self.coarse_earth_mover + model.alpha * self.fine_chamfer

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, model, dataset, snap=None, epoch=None, debug_NaN=False):
        """
        Train the model on a particular dataset.
        """

        if debug_NaN:
            # Add checking ops
            self.check_op = tf.add_check_numerics_ops()

        # Parameters log file
        if model.config.saving:
            model.parameters_log()

        # Save points of the kernel to file
        self.save_kernel_points(model, 0)

        if model.config.saving:
            # Training log file
            if not exists(join(model.saving_path, 'training.txt')):
                with open(join(model.saving_path, 'training.txt'), "w") as file:
                    file.write('epoch steps out_loss reg_loss point_loss coarse_EM fine_CD mixed_loss time memory\n')
            # TODO: delete training.txt lines after snapshot

            # Killing file (simply delete this file when you want to stop the training)
            if not exists(join(model.saving_path, 'running_PID.txt')):
                with open(join(model.saving_path, 'running_PID.txt'), "w") as file:
                    file.write('Launched with PyCharm')

        # Train loop variables
        t0 = time.time()
        self.training_step = 0 if snap is None else snap
        self.training_epoch = 0 if epoch is None else epoch
        mean_dt = np.zeros(2)
        last_display = t0
        self.training_preds = np.zeros(0)
        self.training_labels = np.zeros(0)
        epoch_n = 1
        mean_epoch_n = 0

        # Initialise iterator with train data
        self.sess.run(dataset.train_init_op)

        # Assign hyperparameter alpha only after restoration
        # TODO: implement for generic amount of alpha epoch values
        alpha_idx = 0
        if 0 <= self.training_epoch < model.config.alpha_epoch[1]:
            alpha_idx = 0
        elif model.config.alpha_epoch[1] <= self.training_epoch < model.config.alpha_epoch[2]:
            alpha_idx = 1
        elif model.config.alpha_epoch[2] <= self.training_epoch < model.config.alpha_epoch[3]:
            alpha_idx = 2
        elif self.training_epoch >= model.config.alpha_epoch[3]:
            alpha_idx = 3
        op = model.alpha.assign(model.config.alphas[alpha_idx])
        self.sess.run(op)

        # Start loop
        while self.training_epoch < model.config.max_epoch:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = [self.train_op,
                       model.output_loss,
                       model.regularization_loss,
                       model.offsets_loss,
                       model.coarse,
                       model.complete_points,
                       self.coarse_earth_mover,
                       self.fine_chamfer,
                       self.mixed_loss,
                       model.alpha]

                # If NaN appears in a training, use this debug block
                if debug_NaN:
                    all_values = self.sess.run(ops + [self.check_op] + list(dataset.flat_inputs),
                                               {model.dropout_prob: 0.5})
                    L_out, L_reg, L_p, coarse, complete, coarse_em, fine_cd, mixed_loss, alppha = all_values[1:7]
                    if np.isnan(L_reg) or np.isnan(L_out):
                        input_values = all_values[8:]
                        self.debug_nan(model, input_values, coarse)
                        a = 1 / 0

                else:
                    # Run normal
                    _, L_out, L_reg, L_p, coarse, complete, coarse_em, fine_cd, mixed_loss, alppha = \
                        self.sess.run(ops, {model.dropout_prob: 0.5})

                t += [time.time()]

                # Stack prediction for training confusion
                # if model.config.network_model == 'classification':
                #     self.training_preds = np.hstack((self.training_preds, np.argmax(probs, axis=1)))
                #     self.training_labels = np.hstack((self.training_labels, complete))
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:04d} / Step {:08d} L_out={:5.3f} L_reg={:5.3f} L_p={:5.3f} Coarse_EM={:4.3f} ' \
                              'Fine_CD={:4.3f} Mixed_Loss={:4.3f} alpha={:5.3f} ---{:8.2f} ms/batch (Averaged)'
                    print(message.format(self.training_epoch,
                                         self.training_step,
                                         L_out,
                                         L_reg,
                                         L_p,
                                         coarse_em,
                                         fine_cd,
                                         mixed_loss,
                                         alppha,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

                # Log file
                if model.config.saving:
                    process = psutil.Process(os.getpid())
                    with open(join(model.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {:.1f}\n'
                        file.write(message.format(self.training_epoch,
                                                  self.training_step,
                                                  L_out,
                                                  L_reg,
                                                  L_p,
                                                  coarse_em,
                                                  fine_cd,
                                                  mixed_loss,
                                                  t[-1] - t0,
                                                  process.memory_info().rss * 1e-6))

                # Check kill signal (running_PID.txt deleted)
                if model.config.saving and not exists(join(model.saving_path, 'running_PID.txt')):
                    break

                if model.config.dataset.startswith('ShapeNetPart') or model.config.dataset.startswith(
                        'ModelNet'):
                    if model.config.epoch_steps and epoch_n > model.config.epoch_steps:
                        raise tf.errors.OutOfRangeError(None, None, '')

            except tf.errors.OutOfRangeError:

                # End of train dataset, update average of epoch steps
                mean_epoch_n += (epoch_n - mean_epoch_n) / (self.training_epoch + 1)
                epoch_n = 0
                self.int = int(np.floor(mean_epoch_n))
                model.config.epoch_steps = int(np.floor(mean_epoch_n))
                if model.config.saving:
                    model.parameters_log()

                # Snapshot
                if model.config.saving and (self.training_epoch + 1) % model.config.snapshot_gap == 0:

                    # Tensorflow snapshot
                    snapshot_directory = join(model.saving_path, 'snapshots')
                    if not exists(snapshot_directory):
                        makedirs(snapshot_directory)
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step + 1)

                    # Save points
                    self.save_kernel_points(model, self.training_epoch)

                # Update learning rate
                if self.training_epoch in model.config.lr_decays:
                    op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                               model.config.lr_decays[self.training_epoch]))
                    self.sess.run(op)

                # Increment
                self.training_epoch += 1

                # Update hyper-parameter alpha
                if self.training_epoch in model.config.alpha_epoch:
                    alpha_idx = model.config.alpha_epoch.index(self.training_epoch)
                    op = model.alpha.assign(model.config.alphas[alpha_idx])
                    self.sess.run(op)

                # Validation
                if model.config.network_model == 'completion':
                    self.completion_validation_error(model, dataset)
                else:
                    raise ValueError('No validation method implemented for this network type')

                # Reset iterator on training data
                self.sess.run(dataset.train_init_op)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

            # Increment steps
            self.training_step += 1
            epoch_n += 1

        # Remove File for kill signal
        if exists(join(model.saving_path, 'running_PID.txt')):
            remove(join(model.saving_path, 'running_PID.txt'))
        self.sess.close()

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def completion_validation_error(self, model, dataset):
        """
        Validation method for completion models
        """

        ##########
        # Initiate
        ##########

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over all models
        # TODO: Unused?...
        if not hasattr(self, 'val_dist'):
            self.val_dist = np.zeros((len(dataset.complete_points['valid']), 2))
        # Choose validation smoothing parameter (0 for no smoothing, 0.99 for big smoothing)
        val_smooth = 0.95

        #####################
        # Network predictions
        #####################

        coarse_em_list = []
        fine_cd_list = []
        mixed_loss_list = []
        coarse_list = []
        fine_list = []
        complete_points_list = []
        partial_points_list = []
        obj_inds = []

        mean_dt = np.zeros(2)
        last_display = time.time()
        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.coarse_earth_mover, self.fine_chamfer, self.mixed_loss, model.coarse, model.fine,
                       model.complete_points,
                       model.inputs['points'], model.inputs['object_inds'])
                coarse_em, fine_cd, mixed_loss, coarse, fine, complete, partial, inds = self.sess.run(ops, {
                    model.dropout_prob: 1.0})
                t += [time.time()]

                # Get distances and obj_indexes
                coarse_em_list += [coarse_em]
                fine_cd_list += [fine_cd]
                mixed_loss_list += [mixed_loss]
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
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * len(obj_inds) / model.config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        coarse_em_mean = np.mean(coarse_em_list)
        fine_cd_mean = np.mean(fine_cd_list)
        mixed_loss_mean = np.mean(mixed_loss_list)
        print(
            'Validation distances\nMean (Fine) Chamfer: {:4.3f}\tMean (Coarse) Earth Mover: {:4.3f}\tMean Mixed Loss: '
            '{:4.3f}'.format(
                fine_cd_mean,
                coarse_em_mean,
                mixed_loss_mean))

        if model.config.saving:
            # Validation log file
            if not exists(join(model.saving_path, 'validation.txt')):
                with open(join(model.saving_path, 'validation.txt'), "w") as file:
                    file.write('epoch steps mean_coarse_EM mean_fine_CD mean_mixed_loss\n')
                    message = '{:d} {:d} {:.3f} {:.3f} {:.3f}\n'
                    file.write(message.format(self.training_epoch,
                                              self.training_step,
                                              coarse_em_mean,
                                              fine_cd_mean,
                                              mixed_loss_mean))
            else:
                with open(join(model.saving_path, 'validation.txt'), "a") as file:
                    message = '{:d} {:d} {:.3f} {:.3f} {:.3f}\n'
                    file.write(message.format(self.training_epoch,
                                              self.training_step,
                                              coarse_em_mean,
                                              fine_cd_mean,
                                              mixed_loss_mean))

            if not exists(join(model.saving_path, 'visu', 'valid')):
                makedirs(join(model.saving_path, 'visu', 'valid'))

            all_pcs = [partial_points_list, coarse_list, fine_list, complete_points_list]
            visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']
            for i in range(0, len(coarse_list), 5):
                plot_path = join(model.saving_path, 'visu', 'valid',  # TODO: add ids as plot filename
                                 'epoch_%d_step_%d_%d.png' % (self.training_epoch, self.training_step, i))
                pcs = [x[i] for x in all_pcs]
                partial_temp = pcs[0][0][:model.config.num_input_points, :]
                coarse_temp = pcs[1][0, :, :]
                fine_temp = pcs[2][0, :, :]
                complete_temp = pcs[3][:model.config.num_gt_points, :]
                final_pcs = [partial_temp, coarse_temp, fine_temp, complete_temp]
                self.plot_pc_compare_views(plot_path, final_pcs, visualize_titles)

    # Saving methods
    # ------------------------------------------------------------------------------------------------------------------

    def save_kernel_points(self, model, epoch):
        """
        Method saving kernel point disposition and current model weights for later visualization
        """

        if model.config.saving:

            # Create a directory to save kernels of this epoch
            kernels_dir = join(model.saving_path, 'kernel_points', 'epoch{:d}'.format(epoch))
            if not exists(kernels_dir):
                makedirs(kernels_dir)

            # Get points
            all_kernel_points_tf = [v for v in tf.global_variables() if 'kernel_points' in v.name
                                    and v.name.startswith('KernelPoint')]
            all_kernel_points = self.sess.run(all_kernel_points_tf)

            # Get Extents
            if False and 'gaussian' in model.config.convolution_mode:
                all_kernel_params_tf = [v for v in tf.global_variables() if 'kernel_extents' in v.name
                                        and v.name.startswith('KernelPoint')]
                all_kernel_params = self.sess.run(all_kernel_params_tf)
            else:
                all_kernel_params = [None for p in all_kernel_points]

            # Save in ply file
            for kernel_points, kernel_extents, v in zip(all_kernel_points, all_kernel_params, all_kernel_points_tf):

                # Name of saving file
                ply_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.ply'
                ply_file = join(kernels_dir, ply_name)

                # Data to save
                if kernel_points.ndim > 2:
                    kernel_points = kernel_points[:, 0, :]
                if False and 'gaussian' in model.config.convolution_mode:
                    data = [kernel_points, kernel_extents]
                    keys = ['x', 'y', 'z', 'sigma']
                else:
                    data = kernel_points
                    keys = ['x', 'y', 'z']

                # Save
                write_ply(ply_file, data, keys)

            # Get Weights
            all_kernel_weights_tf = [v for v in tf.global_variables() if 'weights' in v.name
                                     and v.name.startswith('KernelPointNetwork')]
            all_kernel_weights = self.sess.run(all_kernel_weights_tf)

            # Save in numpy file
            for kernel_weights, v in zip(all_kernel_weights, all_kernel_weights_tf):
                np_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.npy'
                np_file = join(kernels_dir, np_name)
                np.save(np_file, kernel_weights)

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
