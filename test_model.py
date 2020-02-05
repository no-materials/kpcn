# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
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
from datasets.ShapeNetV1 import ShapeNetV1Dataset
from datasets.ShapeNetBenchmark2048 import ShapeNetBenchmark2048Dataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def test_caller(path, step_ind, on_val, dataset_path, noise, calc_tsne):
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

    # Adjust batch num if only a single model is to be completed
    if on_val:
        val_data_paths = sorted([os.path.join(dataset_path, 'val', 'partial', k.rstrip() + '.h5')
                                 for k in open(os.path.join(dataset_path, 'val.list').readlines())])
        if int(len(val_data_paths)) == 1:
            config.validation_size = 1
            config.batch_num = 1
    else:
        test_data_paths = sorted([os.path.join(dataset_path, 'test', 'partial', k.rstrip() + '.h5')
                                 for k in open(os.path.join(dataset_path, 'val.list').readlines())])
        if int(len(test_data_paths)) == 1:
            config.validation_size = 1
            config.batch_num = 1

    # Augmentations
    config.augment_scale_anisotropic = True
    config.augment_symmetries = [False, False, False]
    config.augment_rotation = 'none'
    config.augment_scale_min = 1.0
    config.augment_scale_max = 1.0
    config.augment_noise = noise
    config.augment_occlusion = 'none'

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    dl0 = 0  # config.first_subsampling_dl
    if config.dataset.startswith('ShapeNetV1'):
        dataset = ShapeNetV1Dataset()
        # Create subsample clouds of the models
        dataset.load_subsampled_clouds(dl0)
    elif config.dataset.startswith("pc_shapenetCompletionBenchmark2048"):
        dataset = ShapeNetBenchmark2048Dataset(config.batch_num, config.num_input_points, dataset_path)
        # Create subsample clouds of the models
        dataset.load_subsampled_clouds(dl0)  # TODO: careful dl0 is used here - padding?
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Initialize input pipelines
    if on_val:
        dataset.init_input_pipeline(config)
    else:
        dataset.init_test_input_pipeline(config)

    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    if config.dataset.startswith('ShapeNetV1') or config.dataset.startswith("pc_shapenetCompletionBenchmark2048"):
        model = KernelPointCompletionNetwork(dataset.flat_inputs, config, args.double_fold)
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

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

    if config.dataset.startswith('ShapeNetV1') or config.dataset.startswith("pc_shapenetCompletionBenchmark2048"):
        tester.test_completion(model, dataset, on_val, calc_tsne)
    else:
        raise ValueError('Unsupported dataset')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="Test model on the ShapeNetV1 dataset", )
    parser.add_argument('--saving_path', default='last_ShapeNetV1')
    parser.add_argument('--snap', type=int, default=-1, help="snapshot to restore (-1 for latest snapshot)")
    parser.add_argument('--dataset_path')
    parser.add_argument('--on_val', action='store_true')
    parser.add_argument('--double_fold', action='store_true')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--calc_tsne', action='store_true')
    args = parser.parse_args()

    ##########################
    # Choose the model to test
    ##########################

    #
    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_ShapeNetV1': Automatically retrieve the last trained model on ShapeNetV1
    #
    #       > 'results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    #

    chosen_log = args.saving_path

    #
    #   You can also choose the index of the snapshot to load (last by default)
    #

    chosen_snapshot = args.snap

    #
    #   If you want to modify certain parameters in the Config class, for example, to stop augmenting the input data,
    #   there is a section for it in the function "test_caller" defined above.
    #

    ###########################
    # Call the test initializer
    ###########################

    handled_logs = ['last_ShapeNetV1']

    # Automatically retrieve the last trained model
    if chosen_log in handled_logs:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in handled_logs:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    test_caller(chosen_log, chosen_snapshot, args.on_val, args.dataset_path, args.noise, args.calc_tsne)
