# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
import open3d as o3d


def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cat_model_id')
    parser.add_argument('ply_output_dir')  # e.g. /kpcn/data/shapenetV1/train/partial
    parser.add_argument('num_scans', type=int)
    args = parser.parse_args()

    render_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'render_out')

    intrinsics = np.loadtxt(os.path.join(render_output_dir, 'intrinsics.txt'))
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    depth_dir = os.path.join(render_output_dir, 'depth', args.cat_model_id)
    ply_dir = os.path.join(args.ply_output_dir, args.cat_model_id)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)
    for i in range(args.num_scans):
        exr_path = os.path.join(render_output_dir, 'exr', args.cat_model_id, '%d.exr' % i)
        pose_path = os.path.join(render_output_dir, 'pose', args.cat_model_id, '%d.txt' % i)

        depth = read_exr(exr_path, height, width)
        depth_img = o3d.geometry.Image(np.uint16(depth * 1000))
        o3d.io.write_image(os.path.join(depth_dir, '%d.png' % i), depth_img)

        pose = np.loadtxt(pose_path)
        points = depth2pcd(depth, intrinsics, pose)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(ply_dir, '%d.ply' % i), cloud)
