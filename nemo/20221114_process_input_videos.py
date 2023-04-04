"""
for processing anonymized videos
"""
import argparse
import os
import os.path as osp
import yaml
import ipdb
from hmr.video import video_to_images, run_openpose

R = '/home/groups/syyeung/wangkua1/data/mymocap/anonymized'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",
                        type=str,
                        default='nemo/config/mymocap-tennis-swing.yml')
    args = parser.parse_args()

    cfg_path = args.cfg_path
    cfg = yaml.safe_load(open(cfg_path, 'r'))

    # Create experiment dir
    os.makedirs(cfg['exp_dir'], exist_ok=True)

    # Process video paths
    video_fpath_list = []
    for name in cfg['videos']['names']:
        A = name[:-4]
        A = A + '_anonymized'
        name = A + '.mp4'
        video_fpath = osp.join(R, name)
        video_fpath_list.append(video_fpath)
        img_folder = osp.join(cfg['exp_dir'], name + '.anon')
        video_to_images(video_fpath, img_folder)