import os
import concurrent.futures
import subprocess
import logging
import tqdm
import multiprocessing
import argparse

from utils import common

# CONFIG THESE PARAMS --------------------
src_dataset_dir = "/Volumes/warm_blue/datasets/ShapeNetV1"
dataset = "shapenetV1"
split_type = "valid"  # train/valid/test/test_novel
executable = "sample/build/mesh_sampling"


# ----------------------------------------786f18c5f99f7006b1d1509c24a9f631

def process_mesh(mesh_filepath, target_filepath, exe):
    logging.info(mesh_filepath + " --> " + target_filepath)
    additional_args = ["-no_vis_result"]  # additional args here like no vis etc...
    command = [exe, mesh_filepath, target_filepath] + additional_args

    subproc = subprocess.Popen(command)
    subproc.wait()

    logging.info(target_filepath + " complete point cloud generated using uniform sampling.")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source in order to generate "
                    "complete uniformly sampled point clouds",
    )
    common.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    common.configure_logging(args)

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_list_file = os.path.join(root_dir, 'data', dataset, '%s.list' % split_type)
    target_data_dir = os.path.join(root_dir, 'data', dataset, split_type, 'complete')
    num_threads = multiprocessing.cpu_count()

    with open(model_list_file) as file:
        model_list = file.read().splitlines()
        file.close()

    for i, cat_model_id in tqdm.tqdm(enumerate(model_list)):

        cat, model_id = cat_model_id.split('/')
        if not os.path.isdir(os.path.join(target_data_dir, cat)):
            os.makedirs(os.path.join(target_data_dir, cat))

        target_mesh_file = os.path.join(target_data_dir, cat, '%s.ply' % model_id)
        if not os.path.isfile(target_mesh_file):
            mesh_src_file = os.path.join(src_dataset_dir, cat, model_id, 'model.obj')
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(num_threads)) as executor:
                executor.submit(
                    process_mesh,
                    mesh_src_file,
                    target_mesh_file,
                    executable
                )

            executor.shutdown()
