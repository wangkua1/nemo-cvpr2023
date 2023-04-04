import argparse
import os
import os.path as osp
import yaml
import ipdb
from hmr.video import run_openpose

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_fol",
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
        video_fpath = osp.join(cfg['videos']['root_dir'], name)
        video_fpath_list.append(video_fpath)
        img_folder = osp.join(cfg['exp_dir'], name)
        video_to_images(video_fpath, img_folder)
        run_openpose(img_folder, img_folder + '_openpose')
