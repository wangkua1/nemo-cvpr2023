# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp


# def run_openpose(
#         video_file,
#         output_folder,
#         staf_folder,
#         vis=False,
# ):
#     pwd = os.getcwd()

#     os.chdir(staf_folder)

#     render = 1 if vis else 0
#     display = 2 if vis else 0
#     cmd = [
#         'build/examples/openpose/openpose.bin',
#         '--model_pose', 'BODY_21A',
#         '--tracking', '1',
#         '--render_pose', str(render),
#         '--video', video_file,
#         '--write_json', output_folder,
#         '--display', str(display)
#     ]

#     print('Executing', ' '.join(cmd))
#     subprocess.call(cmd)
#     os.chdir(pwd)

def run_openpose(
        video_file,
        output_folder,
        staf_folder,
        vis=False,
):
    TMP_DIR = osp.split(video_file)[0]

    # Move video to tmp dir
    video_fname = osp.split(video_file)[1]
    out_video_path = osp.join(TMP_DIR, video_fname)
    subprocess.run(['cp', video_file, out_video_path])
    subprocess.run(['mkdir', osp.join(TMP_DIR, video_fname+'_results')])

    run_cmds = [
        'singularity', 'exec', '--pwd', 'openpose', '--nv', '--bind',
        f"{TMP_DIR}:/mnt",
        "/home/groups/syyeung/wangkua1/software/openpose.sif",
        "./build/examples/openpose/openpose.bin",
        "--video", f"/mnt/{video_fname}",
        "--write_json", f"/mnt/{video_fname+'_results'}",
        "--display", "0",
        '--render_pose', '0',
        '--tracking', '1',
        '--model_pose', 'BODY_25',
        '--number_people_max', '1'
    ]
    print(run_cmds)
    subprocess.run(run_cmds)
    # Copy results to output_folder
    subprocess.run(['mkdir', '-p', output_folder])
    subprocess.run(['cp', '-r', osp.join(TMP_DIR, video_fname+'_results'), output_folder])


def read_posetrack_keypoints(output_folder):

    people = dict()

    # ....
    child_folder = os.listdir(output_folder)[0]
    output_folder = osp.join(output_folder, child_folder)

    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = osp.join(output_folder, result_file)
        data = json.load(open(json_file))
        # print(idx, data)
        for person in data['people']:
            person_id = person['person_id'][0]
            joints2d  = person['pose_keypoints_2d']
            if person_id in people.keys():
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)
            else:
                people[person_id] = {
                    'joints2d': [],
                    'frames': [],
                }
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)

    for k in people.keys():
        people[k]['joints2d'] = np.array(people[k]['joints2d']).reshape((len(people[k]['joints2d']), -1, 3))
        people[k]['frames'] = np.array(people[k]['frames'])

    return people


def run_posetracker(video_file, staf_folder, posetrack_output_folder='/tmp', display=False):
    posetrack_output_folder = os.path.join(
        posetrack_output_folder,
        f'{os.path.basename(video_file)}_posetrack'
    )

    # run posetrack on video
    run_openpose(
        video_file,
        posetrack_output_folder,
        vis=display,
        staf_folder=staf_folder
    )

    people_dict = read_posetrack_keypoints(posetrack_output_folder)

    shutil.rmtree(posetrack_output_folder)

    return people_dict