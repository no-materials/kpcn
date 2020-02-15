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
import open3d as o3d

# Metrics
from utils.metrics import chamfer, earth_mover, minimal_matching_distance

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


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

    def test_completion(self, model, dataset, on_val, calc_tsne, num_votes=100):

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
        latent_feat_list = []
        mmd_list = []

        if on_val:
            self.sess.run(dataset.val_init_op)
            cardinal = dataset.num_valid
            while True:
                try:
                    # Run one step of the model.
                    t = [time.time()]
                    ops = (self.coarse_earth_mover, self.fine_chamfer, model.coarse, model.fine, model.complete_points,
                           model.inputs['points'], model.inputs['object_inds'], model.inputs['ids'],
                           model.bottleneck_features)
                    coarse_em, fine_cd, coarse, fine, complete, partial, inds, idss, latent_feat = self.sess.run(ops, {
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
                    latent_feat_list += [latent_feat]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Test : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(100 * len([item for sublist in obj_inds for item in sublist]) / cardinal,
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
                if not exists(join(model.saving_path, 'visu', 'test_on_val')):
                    makedirs(join(model.saving_path, 'visu', 'test_on_val'))

                all_pcs = [partial_points_list, coarse_list, fine_list, complete_points_list]
                all_dist = [coarse_em_list, fine_cd_list]
                visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']
                for i, id_batch_np in enumerate(ids_list):
                    plot_path = join(model.saving_path, 'visu', 'test_on_val',
                                     '%s.png' % id_batch_np[0].decode().split(".")[0])
                    if not exists(dirname(plot_path)):
                        makedirs(dirname(plot_path))
                    pcs = [x[i] for x in all_pcs]
                    dists = [d[i] for d in all_dist]
                    suptitle = 'Coarse EMD = {:4.5f}    Fine CD = {:4.5f}'.format(dists[0], dists[1])
                    partial_temp = pcs[0][0][:model.config.num_input_points, :]
                    coarse_temp = pcs[1][0, :, :]
                    fine_temp = pcs[2][0, :, :]
                    complete_temp = pcs[3][:model.config.num_gt_points, :]
                    final_pcs = [partial_temp, coarse_temp, fine_temp, complete_temp]
                    self.plot_pc_compare_views(plot_path, final_pcs, visualize_titles, suptitle=suptitle)

            if calc_tsne:
                # t-SNE plot (use PCA_50_dims first for dim reduction)
                features_np = np.array(latent_feat_list)
                features = np.reshape(features_np, (features_np.shape[0] * features_np.shape[1], -1))  # 50 * 16, 2048

                ids_np = np.concatenate(ids_list, axis=None)
                category_ids = np.array([dataset.synset_to_category[id_el.decode().split("/")[0]] for id_el in ids_np])

                df = pd.DataFrame(features)
                df['y'] = category_ids
                pca_50 = PCA(n_components=50)
                pca_result_50 = pca_50.fit_transform(features)
                print('Cumulative explained variation for 50 principal components: {}'.format(
                    np.sum(pca_50.explained_variance_ratio_)))

                time_start = time.time()
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
                tsne_pca_results = tsne.fit_transform(pca_result_50)
                print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
                df['tsne-pca50-one'] = tsne_pca_results[:, 0]
                df['tsne-pca50-two'] = tsne_pca_results[:, 1]

                plt.figure(figsize=(9, 9))
                scatterplot = sns.scatterplot(
                    x="tsne-pca50-one", y="tsne-pca50-two",
                    hue="y",
                    palette=sns.color_palette("hls", 8),
                    data=df.loc[:, :],
                    legend="full",
                    alpha=0.3,
                )
                fig = scatterplot.get_figure()
                fig.savefig('PCA50_tsne_val.png')
                plt.close(fig)

        else:  # on test set

            self.minimal_matching_dist = minimal_matching_distance(model.fine, dataset)

            self.sess.run(dataset.test_init_op)
            cardinal = dataset.num_test
            print("Retrieving {} models from db of size {}".format(cardinal, dataset.num_train))
            retrieval_time_start = time.time()
            while True:
                try:
                    # Run one step of the model.
                    t = [time.time()]
                    ops = (model.coarse, model.fine, model.inputs['points'], model.inputs['object_inds'],
                           model.inputs['ids'], model.bottleneck_features, self.minimal_matching_dist)
                    coarse, fine, partial, inds, idss, latent_feat, mmd = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    coarse_list += [coarse]
                    fine_list += [fine]
                    partial_points_list += [partial]
                    obj_inds += [inds]
                    ids_list += [idss]
                    latent_feat_list += [latent_feat]
                    mmd_list += [mmd]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Test : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(100 * len([item for sublist in obj_inds for item in sublist]) / cardinal,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            print('Retrieval from db done! Time elapsed: {} seconds'.format(time.time() - retrieval_time_start))
            #  mmd_list has shape: [([idx1, idx2, idx3, ..., idx16], [cd1, cd2, cd3, ..., cd16]), (...),...]
            matched_models_list = []
            mmds = []
            for idxb, pair in enumerate(mmd_list):
                for idd in pair[0]:
                    matched_models_list.append(dataset.complete_points['valid'][idd])  # store matched model
                for mmd in pair[1]:
                    mmds.append(mmd)  # store mmd

            matched_models_list = np.array(matched_models_list)
            matched_models_list = np.reshape(matched_models_list,
                                             (-1, dataset.batch_num, matched_models_list.shape[1],
                                              matched_models_list.shape[2]))
            mmds = np.array(mmds)  # shape: (800,)
            mmd_mean = np.mean(mmds)
            print('Test MMD: {:4.5f}'.format(mmd_mean))
            mmds = np.reshape(mmds, (-1, dataset.batch_num))

            if model.config.saving:
                if not exists(join(model.saving_path, 'visu', 'test_retrieval')):
                    makedirs(join(model.saving_path, 'visu', 'test_retrieval'))

                all_pcs = [partial_points_list, coarse_list, fine_list, matched_models_list]
                all_dist = [mmds]
                visualize_titles = ['input', 'coarse output', 'fine output', 'matched model']
                for i, id_batch_np in enumerate(ids_list):
                    plot_path = join(model.saving_path, 'visu', 'test_retrieval',
                                     '%s.png' % id_batch_np[0].decode().split(".")[0])
                    if not exists(dirname(plot_path)):
                        makedirs(dirname(plot_path))
                    pcs = [x[i] for x in all_pcs]
                    dists = [d[i] for d in all_dist]
                    suptitle = 'Minimal Matching Distance (MMD) = {:4.5f}'.format(dists[0][0])
                    partial_temp = pcs[0][0][:model.config.num_input_points, :]
                    coarse_temp = pcs[1][0, :, :]
                    fine_temp = pcs[2][0, :, :]
                    matched_temp = pcs[3][0, :, :]
                    final_pcs = [partial_temp, coarse_temp, fine_temp, matched_temp]
                    self.plot_pc_compare_views(plot_path, final_pcs, visualize_titles, suptitle=suptitle)

            if calc_tsne:
                # t-SNE plot (use PCA_50_dims first for dim reduction)
                features_np = np.array(latent_feat_list)
                features = np.reshape(features_np, (features_np.shape[0] * features_np.shape[1], -1))  # 50 * 16, 2048

                df = pd.DataFrame(features)
                pca_50 = PCA(n_components=50)
                pca_result_50 = pca_50.fit_transform(features)
                print('Cumulative explained variation for 50 principal components: {}'.format(
                    np.sum(pca_50.explained_variance_ratio_)))

                time_start = time.time()
                tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
                tsne_pca_results = tsne.fit_transform(pca_result_50)
                print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
                df['tsne-pca50-one'] = tsne_pca_results[:, 0]
                df['tsne-pca50-two'] = tsne_pca_results[:, 1]

                plt.figure(figsize=(9, 9))
                scatterplot = sns.scatterplot(
                    x="tsne-pca50-one", y="tsne-pca50-two",
                    palette=sns.color_palette("hls", 8),
                    data=df.loc[:, :],
                    legend="full",
                    alpha=0.3,
                )
                fig = scatterplot.get_figure()
                fig.savefig('PCA50_tsne_test.png')
                plt.close(fig)

        return

    def test_kitti_completion(self, model, dataset, shapenet2048_dataset):
        # Set MMD metric op by passing ShapeNet dataset
        self.minimal_matching_dist = minimal_matching_distance(model.fine, shapenet2048_dataset)

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
        mmd_list = []

        while True:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (
                    model.coarse, model.fine, model.inputs['points'], model.inputs['object_inds'], model.inputs['ids'],
                    self.minimal_matching_dist)
                coarse, fine, partial, inds, idss, mmd = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get results and append to list
                coarse_list += [coarse]
                fine_list += [fine]
                partial_points_list += [partial]
                obj_inds += [inds]
                ids_list += [idss]
                mmd_list += [mmd]

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

        # Gather mmd and respective matched model from mmd_list, also calc mean mmd
        #  mmd_list has shape: [([idx1, idx2, idx3, ..., idx16], [cd1, cd2, cd3, ..., cd16]), (...),...]
        matched_models_list = []
        mmds = []
        for idxb, pair in enumerate(mmd_list):
            # for idd in pair[0]:
            #     matched_models_list.append(shapenet2048_dataset.complete_points['valid'][idd])  # store matchd model
            for mmd in pair[1]:
                mmds.append(mmd)  # store mmd

        # matched_models_list = np.array(matched_models_list)
        # matched_models_list = np.reshape(matched_models_list,
        #                                  (-1, dataset.batch_num, matched_models_list.shape[1],
        #                                   matched_models_list.shape[2]))
        mmds = np.array(mmds)  # shape: (2401,)
        mmd_mean = np.mean(mmds)
        print('Test MMD: {:4.5f}'.format(mmd_mean))
        mmds = np.reshape(mmds[:-1], (-1, dataset.batch_num))

        if model.config.saving:
            if not exists(join(model.saving_path, 'visu', 'kitti', 'plots')):
                makedirs(join(model.saving_path, 'visu', 'kitti', 'plots'))

            if not exists(join(model.saving_path, 'visu', 'kitti', 'completions')):
                makedirs(join(model.saving_path, 'visu', 'kitti', 'completions'))

            # Plot & save completed pcd code
            all_pcs = [partial_points_list, coarse_list, fine_list]
            all_dist = [mmds]
            visualize_titles = ['input', 'coarse output', 'fine output']
            for i, id_batch_np in enumerate(ids_list):
                for j, batch_el_idx in enumerate(id_batch_np):

                    car_id = id_batch_np[j].decode().split(".")[0]
                    pcs = [x[i] for x in all_pcs]
                    dists = [d[i] for d in all_dist]
                    suptitle = 'Minimal Matching Distance (MMD) = {:4.5f}'.format(dists[0][0])

                    # Plot
                    if j == 0:
                        plot_path = join(model.saving_path, 'visu', 'kitti', 'plots', '%s.png' % car_id)
                        if not exists(dirname(plot_path)):
                            makedirs(dirname(plot_path))
                        partial_temp = pcs[0][0][:model.config.num_input_points, :]
                        coarse_temp = pcs[1][0, :, :]
                        fine_temp = pcs[2][0, :, :]
                        final_pcs = [partial_temp, coarse_temp, fine_temp]
                        self.plot_pc_compare_views(plot_path, final_pcs, visualize_titles, suptitle=suptitle)

                    # Save pcd
                    # Calculate center, rotation and scale
                    bbox = np.loadtxt(join(dataset.bbox_dir, '%s.txt' % car_id))
                    center = (bbox.min(0) + bbox.max(0)) / 2
                    bbox -= center
                    yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
                    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                         [np.sin(yaw), np.cos(yaw), 0],
                                         [0, 0, 1]])
                    bbox = np.dot(bbox, rotation)
                    scale = bbox[3, 0] - bbox[0, 0]
                    bbox /= scale

                    completion_w = np.dot(pcs[2][j, :, :], [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                    completion_w = np.dot(completion_w * scale, rotation.T) + center
                    pcd_path = join(model.saving_path, 'visu', 'kitti', 'completions', '%s.pcd' % car_id)
                    if not exists(dirname(pcd_path)):
                        makedirs(dirname(pcd_path))
                    self.save_pcd(pcd_path, completion_w)

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

    @staticmethod
    def save_pcd(filename, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)
