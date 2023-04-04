import argparse
import os
import os.path as osp
import yaml
import ipdb
from hmr.video import run_openpose
from hmr.penn_action import PENN_ACTION_ROOT
import pandas


if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seq_name", type=str)
    # args = parser.parse_args()

    label_name = 'penn_action_leftright_label_20220519.csv'
    label_file = osp.join(PENN_ACTION_ROOT, label_name)
    df = pandas.read_csv(label_file)
    seq_names = [name[-4:] for name in list(df['seq_name'])]

    for seq_name in seq_names:
        output_dir = osp.join(PENN_ACTION_ROOT, 'openpose', seq_name)
        img_folder = osp.join(PENN_ACTION_ROOT, 'frames', seq_name)
        run_openpose(img_folder, output_dir)
