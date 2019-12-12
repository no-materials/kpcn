# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on kitti raw data
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import numpy as np
import argparse

# My libs
from utils.config import Config
from utils.tester import ModelTester
from models.KPCN_model import KernelPointCompletionNetwork

# Datasets
from datasets.kitti import KittiDataset


def test_caller(path, step_ind, dataset_path):
    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 0.0001
    # config.augment_color = 1.0
    # config.validation_size = 500
    # config.batch_num = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    dataset = KittiDataset(config.batch_num, config.num_input_points, dataset_path)

    # Create subsample clouds of the models
    dl0 = 0  # config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)

    # Initialize test input pipeline
    dataset.init_test_input_pipeline(config)

    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    model = KernelPointCompletionNetwork(dataset.flat_inputs, config, args.double_fold)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    if step_ind == -1:
        chosen_step = np.sort(snap_steps)[step_ind]
    else:
        chosen_step = step_ind + 1
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    tester = ModelTester(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')
    tester.test_kitti_completion(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saving_path', help="model_log_file_path")
    parser.add_argument('--snap', type=int, default=-1, help="snapshot to restore (-1 for latest snapshot)")
    parser.add_argument('--dataset_path')
    args = parser.parse_args()

    chosen_log = args.saving_path
    chosen_snapshot = args.snap

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    test_caller(chosen_log, chosen_snapshot, args.dataset_path)
