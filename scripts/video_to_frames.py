import os
import os.path as osp
import subprocess
import cv2
from tqdm import tqdm
from itertools import product

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
	for action, index in tqdm(product(['baseball_pitch', 'baseball_swing', 'golf_swing', 'tennis_swing', 'tennis_serve'], range(8))):
		vid_name = f"{action}.{index}.mp4"
		video_to_images(osp.join('data/videos', vid_name), osp.join('data/exps', f'mymocap_{action}', vid_name))


if __name__ == '__main__':
	main()