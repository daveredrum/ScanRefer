import os
import sys
import time
import math
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import data.scannet.scannet_utils as scannet_utils

from lib.config import CONF
from lib.projection import ProjectionHelper
from lib.utils import get_eta

# data path
SCANNET_LIST = CONF.SCANNETV2_LIST
SCANNET_DATA = CONF.PATH.SCANNET_DATA
PROJECTION_ROOT = CONF.PROJECTION
PROJECTION_PATH = os.path.join(PROJECTION_ROOT, "{}_{}.npy") # scene_id, mode

# scannet data
# NOTE: read only!
SCANNET_FRAME_ROOT = CONF.SCANNET_FRAMES
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAME_ROOT, "{}") # name of the file

# projection
INTRINSICS = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [41, 32], 0.05)

def get_scene_list():
    with open(SCANNET_LIST, 'r') as f:
        scene_list = sorted(list(set(f.read().splitlines())))
    
    return scene_list

def load_scene(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        # load the original vertices, not the axis-aligned ones
        scene_data[scene_id] = np.load(os.path.join(SCANNET_DATA, scene_id)+"_vert.npy")[:, :3]

    return scene_data

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims != new_image_dims:
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    
    return np.array(image)

def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]

    return np.asarray(lines).astype(np.float32)

def load_image(file, image_dims):
    image = imread(file)
    # preprocess
    image = resize_crop_image(image, image_dims)
    if len(image.shape) == 3: # color image
        image =  np.transpose(image, [2, 0, 1])  # move feature to front
        image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
    elif len(image.shape) == 2: # label image
        image = np.expand_dims(image, 0)
    else:
        raise ValueError

    return image

def load_depth(file, image_dims):
    depth_image = imread(file)
    # preprocess
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0

    return depth_image

def to_tensor(arr):
    return torch.Tensor(arr).cuda()

def compute_projection(points, depth, camera_to_world):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        
        :return indices_3d (array with point indices that correspond to a pixel),
        :return indices_2d (array with pixel indices that correspond to a point)

        note:
            the first digit of indices represents the number of relevant points
            the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
    indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

    for i in range(num_frames):
        indices = PROJECTOR.compute_projection(to_tensor(points), to_tensor(depth[i]), to_tensor(camera_to_world[i]))
        if indices:
            indices_3ds[i] = indices[0].long()
            indices_2ds[i] = indices[1].long()
            print("found {} projections from frame {} to {} points".format(indices_3ds[i][0], i, num_points))
        else:
            print("skipped frame {}".format(i))

    return indices_3ds, indices_2ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # make dir
    os.makedirs(PROJECTION_ROOT, exist_ok=True)
    scene_list = get_scene_list()

    print("loading scene files...")
    scene_data = load_scene(scene_list)

    print("computing multiview projections...")
    for idx, scene_id in enumerate(scene_list):
        # if os.path.exists(PROJECTION_PATH.format(scene_id, "3d")) and os.path.exists(PROJECTION_PATH.format(scene_id, "2d")):
        #     print("skipping {}...".format(scene_id))
        #     continue

        print("processing {}...".format(scene_id))
        start = time.time()

        point_cloud = scene_data[scene_id]
        frame_list = list(map(lambda x: x.split(".")[0], os.listdir(SCANNET_FRAME_ROOT.format(scene_id, "color"))))

        # load frames
        scene_images = np.zeros((len(frame_list), 3, 256, 328))
        scene_depths = np.zeros((len(frame_list), 32, 41))
        scene_poses = np.zeros((len(frame_list), 4, 4))
        for i, frame_id in enumerate(frame_list):
            scene_images[i] = load_image(SCANNET_FRAME_PATH.format(scene_id, "color", "{}.jpg".format(frame_id)), [328, 256])
            scene_depths[i] = load_depth(SCANNET_FRAME_PATH.format(scene_id, "depth", "{}.png".format(frame_id)), [41, 32])
            scene_poses[i] = load_pose(SCANNET_FRAME_PATH.format(scene_id, "pose", "{}.txt".format(frame_id)))

        projection_3d, projection_2d = compute_projection(point_cloud[:, :3], scene_depths, scene_poses)
        
        np.save(PROJECTION_PATH.format(scene_id, "3d"), projection_3d.cpu().numpy())
        np.save(PROJECTION_PATH.format(scene_id, "2d"), projection_2d.cpu().numpy())

        # verbose
        num_left = len(scene_list) - idx - 1
        eta = get_eta(start, time.time(), 0, num_left)
        print("processed {}, {} scenes left, ETA: {}h {}m {}s".format(
            scene_id,
            num_left,
            eta["h"],
            eta["m"],
            eta["s"]
        ))

    print("done!")
        
