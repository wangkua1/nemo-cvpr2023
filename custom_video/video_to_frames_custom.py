import os
import os.path as osp
import subprocess
import cv2
from tqdm import tqdm
from itertools import product
import argparse

# accept command line arguments, including action name and number of videos
parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str, default='tennis_swing')
parser.add_argument('--num_videos', type=int, default=6)
args = parser.parse_args()

def video_to_images(vid_file, img_folder=None, return_info=False):
    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def main():
	for action, index in tqdm(product([args.action], range(args.num_videos))): # replace with your own actions and indices
		vid_name = f"{action}.{index}.mp4"
		output_folder = f"{action}.{index}.frames"
		video_to_images(osp.join('data/videos/', vid_name), osp.join('data/exps/', output_folder))

if __name__ == '__main__':
	main()