import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def chamfer(pcd1, pcd2):
    # return 2
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2


def earth_mover(pcd1, pcd2):
    # return 2
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)


def minimal_matching_distance(pcd_fine, dataset):
    cd_gt_from_fine_list = []

    gt_list = dataset.complete_points['train']

    batch_min_cds_and_gt = []
    for i in range(dataset.batch_num):
        for gt in gt_list:
            cd_gt_from_fine_list += [
                chamfer(tf.expand_dims(pcd_fine[i, :, :], 0), tf.expand_dims(tf.cast(gt, tf.float32), 0))]
        stacked_cds = tf.stack(cd_gt_from_fine_list)
        min_idx = tf.math.argmin(stacked_cds)
        batch_min_cds_and_gt.append(tuple((min_idx, tf.gather(stacked_cds, min_idx))))
        cd_gt_from_fine_list = []

    return batch_min_cds_and_gt
