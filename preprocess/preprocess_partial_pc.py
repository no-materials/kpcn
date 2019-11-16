import concurrent.futures
import subprocess
import logging
import os
import multiprocessing
import argparse

from utils import common

# CONFIG THESE PARAMS --------------------
src_dataset_dir = "/Volumes/warm_blue/datasets/ShapeNetV1"
dataset = "shapenetV1"
split_type = "test"  # train/valid/test/test_novel
num_scans = 2
blender_path = "/Volumes/warm_blue/blender-git/build_darwin_full/bin/Blender.app/Contents/MacOS/Blender"


# ----------------------------------------

def simulate_depth_scan(cat_model_id, target_mesh_dir):
    logging.info(cat_model_id + " --> " + target_mesh_dir + " : " + str(num_scans) + " scans")

    # simulate depth scan via blender virtual render
    command = [blender_path,
               '-b',
               '-P',
               os.path.join(os.path.dirname(os.path.realpath(__file__)), 'render', "render_depth.py"),
               src_dataset_dir,
               cat_model_id,
               render_out_dir,
               str(num_scans)]

    subproc_render = subprocess.Popen(command)
    subproc_render.wait()

    # generate_partial_point_cloud
    command = ['python',
               os.path.join(os.path.dirname(os.path.realpath(__file__)), 'render', "process_exr.py"),
               cat_model_id,
               target_data_dir,
               str(num_scans)]

    subproc_gen_pc = subprocess.Popen(command)
    subproc_gen_pc.wait()

    logging.info(target_mesh_dir + " partial point cloud generated.")


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source in order to generate "
                    "partial point clouds via virtual depth rendering",
    )
    common.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    common.configure_logging(args)

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_list_file = os.path.join(root_dir, 'data', dataset, '%s.list' % split_type)
    target_data_dir = os.path.join(root_dir, 'data', dataset, split_type, 'partial')
    render_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'render', "render_out")
    num_threads = multiprocessing.cpu_count()

    with open(model_list_file) as file:
        model_list = file.read().splitlines()
        file.close()

    # if os.path.isdir(render_out_dir):
    #     os.rmdir(render_out_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(num_threads)) as executor:
        for i, cat_model_id in enumerate(model_list):

            cat, model_id = cat_model_id.split('/')
            if not os.path.isdir(os.path.join(target_data_dir, cat)):
                os.makedirs(os.path.join(target_data_dir, cat))

            target_mesh_dir = os.path.join(target_data_dir, cat, model_id)
            if not os.path.isdir(target_mesh_dir):
                os.makedirs(target_mesh_dir)

            # Check if num_scans matches with existing num of files in target dir - if not, only then preprocess
            # TODO: fix case where num scan < num of plys in mesh dir
            if num_scans != len([f for f in os.listdir(target_mesh_dir)
                                 if f.endswith('.ply') and os.path.isfile(os.path.join(target_mesh_dir, f))]):
                mesh_src_file = os.path.join(src_dataset_dir, cat, model_id, 'model.obj')

                executor.submit(
                    simulate_depth_scan,
                    cat_model_id,
                    target_mesh_dir
                )

        executor.shutdown()
