from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms
#from .util import Equirec2Cube



def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Stanford2D3D(data.Dataset):
    """The Stanford2D3D Dataset"""

    def __init__(self, root_dir, list_file, height=256, width=512, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        if is_training:
            self.rgb_depth_list = 10 * self.rgb_depth_list
        self.w = width
        self.h = height
        #self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)
        self.max_depth_meters = 10.0
        self.pilToTensor =  transforms.ToTensor()
        
        self.color_augmentation = False
        self.LR_filp_augmentation = False
        self.yaw_rotation_augmentation = False

        self.is_training = is_training

        #self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        #gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        #sf2d3d
        #gt_depth = cv2.resize(gt_depth, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
        #gt_depth = gt_depth.astype(np.float)/512
        
        #shanghai360
        gt_depth = cv2.resize(gt_depth, dsize=(512, 256), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float)/325
        gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters 
        gt_depth[gt_depth < 0] = 0



        
        aug_rgb = rgb

        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        #cube_rgb = self.e2c.run(rgb)
        #cube_aug_rgb = self.e2c.run(aug_rgb)

        #rgb = self.to_tensor(rgb.copy())
        #cube_rgb = self.to_tensor(cube_rgb.copy())
        #aug_rgb = self.to_tensor(aug_rgb.copy())
        #cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())

        inputs["rgb"] = self.pilToTensor(rgb)
        #inputs["leftRGB"] = self.normalize(aug_rgb)

        #inputs["cube_rgb"] = self.pilToTensor(cube_rgb)
        #inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)


        inputs["leftDepth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        #inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)& ~torch.isnan(inputs["gt_depth"]))

        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """

        return inputs



