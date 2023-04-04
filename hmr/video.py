import os
import sys
import shutil
import argparse
import subprocess
import time
import json
import glob

from multiprocessing import Pool

import os.path as osp
import torch, torchvision
from PIL import Image
from torchvision import transforms
import numpy as np

import cv2


def video_to_images(vid_file,
                    img_folder=None,
                    return_info=False,
                    force_run=False):
    if not force_run and osp.exists(img_folder):
        print(f'>>> The follow image_folder exists, skipping: {img_folder}')
    else:
        os.makedirs(img_folder, exist_ok=True)

        command = [
            'ffmpeg', '-i', vid_file, '-f', 'image2', '-v', 'error',
            f'{img_folder}/%06d.png'
        ]
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)

        print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder



def make_absolute(rel_paths):
    ''' Makes a list of relative paths absolute '''
    return [os.path.join(os.getcwd(), rel_path) for rel_path in rel_paths]


SKELETON = 'BODY_25'


def run_openpose(img_dir, out_dir, video_out=None, img_out=None, render=False):
    '''
    Runs OpenPose for 2D joint detection on the images in img_dir.
    '''
    # make all paths absolute to call OP
    img_dir = make_absolute([img_dir])[0]
    out_dir = make_absolute([out_dir])[0]
    # if video_out is not None:
    #     video_out = make_absolute([video_out])[0]
    # if img_out is not None:
    #     img_out = make_absolute([img_out])[0]

    # if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if img_out is not None:
        os.makedirs(osp.join(img_dir, img_out), exist_ok=True)

    # run open pose

    run_cmds = [
        'singularity', 'exec', '--pwd', 'openpose', '--nv', '--bind',
        f"{img_dir}:/mnt",
        "/home/groups/syyeung/wangkua1/software/openpose.sif",
        "./build/examples/openpose/openpose.bin", "--image_dir", "/mnt/.",
        "--write_json", "/mnt/keypoints", "--display", "0", '--model_pose',
        SKELETON, '--number_people_max', '1'
    ]

    if video_out is not None:
        run_cmds += ['--write_video', osp.join('/mnt', video_out), '--write_video_fps', '30']
    if img_out is not None:
        run_cmds += ['--write_images', osp.join('/mnt', img_out)]
    if not (video_out is not None or img_out is not None):
        run_cmds += ['--render_pose', '0']
    print(run_cmds)
    subprocess.run(run_cmds)

    # Copy JSON result to out_dir
    for file in os.listdir(os.path.join(img_dir, 'keypoints')):
        src = os.path.join(img_dir, 'keypoints', file)
        shutil.copyfile(src, os.path.join(out_dir, file))