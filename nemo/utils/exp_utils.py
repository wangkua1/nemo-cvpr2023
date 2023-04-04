import os
import os.path as osp
import numpy as np
import yaml
from copy import deepcopy
from time import time
from types import SimpleNamespace


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time()
        dur = self.t1 - self.t0
        print('')
        print("Timer >>> ")
        print(f"{self.name} took: {dur:.2f} seconds")
        print('')


def find_latest_ckpt(ckpt_dir):
    if not osp.exists(ckpt_dir):
        return ''

    all_ckpts = os.listdir(ckpt_dir)
    # No child dir
    if all_ckpts == []:
        ckpt_name = ''
    else:
        ckpt_name = sorted(all_ckpts)[-1]
    return ckpt_name

def find_latest_child_dir_id(exp_dir):
    if not osp.exists(exp_dir):
        return -1

    all_child_dirs = os.listdir(exp_dir)
    # No child dir
    if all_child_dirs == []:
        latest_child_dir_id = -1
    else:  # if you get an error here, likely it's b/c the folder contains other things.
        latest_child_dir_id = max(map(lambda s: int(s), all_child_dirs))
    return latest_child_dir_id


def create_latest_child_dir(exp_dir):
    cur_child_dir_id = find_latest_child_dir_id(exp_dir) + 1
    child_dir_path = osp.join(exp_dir, f"{cur_child_dir_id:06d}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(child_dir_path, exist_ok=True)
    return child_dir_path


def process_default_config(parser, cmdline_keys):
    args = parser.parse_args()
    input_args = vars(args)
    script_defaults = vars(parser.parse_args([]))
    if 'default_config' in input_args and input_args['default_config']:
        input_defaults = yaml.safe_load(open(input_args['default_config'],
                                             'r'))
        config = deepcopy(script_defaults)

        # Overwrite with `input_defaults` from yml file
        for k in input_defaults:
            config[k] = input_defaults[k]

        # Overwrite with `input_args` from command line
        for k in input_args:
            # Only if the input is in cmdline_keys (i.e. specified in cmdline)
            if k in cmdline_keys:
                config[k] = input_args[k]
        args = SimpleNamespace(**config)
        return args
    else:
        return args
