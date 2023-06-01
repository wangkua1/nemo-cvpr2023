# Running NeMo [CVPR2023 Highlight] On Custom Video Data-set

This document will guide you through the procedure of running NeMo on your own multi-view action video dataset.

## Installation

All detailed instructions on configuring the conda environment as well as additional software downloads are specified in the README.md file.

## Description

To run NeMo, we first need to pre-process your custom video file to prepare three directories.

1. Individual frames of the video file
2. OpenPose 2D ground-truth labels on each frame
3. VIBE 3D HMR initial results for each frame

The three subsequent sections will provide detailed instructions for each of the three requirements.

## Converting mp4 video to frames

After placing your mp4 video files within the directory 'data/videos/', open 'scripts/video_to_frames_jeff.py' to insert your video directory name and your action name accordingly in the main() method.

Afterwards, run the following command:

```bash
python -m scripts.video_to_frames_jeff
```

If this is correctly done, you will have a directory 'data/exps/your_experiment_directory/{action}.{index}.mp4' containing the frames of your video. Replace .mp4 with .frames.

## Running OpenPose

Run the command below after inserting the appropriate string in the {}s.

```bash
docker run --gpus 5 --rm -v /{your_cluster_group_directory}/nemo-cvpr2023/data/exps/{your_experiment_directory}:/mnt cwaffles/openpose ./build/examples/openpose/openpose.bin --image_dir /mnt/{action}.{index}.frames --write_json /mnt/{action}.{index}.op --display 0  --model_pose BODY_25 --number_people_max 1 --render_pose 0

```

If you are unable to run docker, you can directly follow the instructions on running OpenPose demo on your own video on the OpenPose github repo.
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

## Running VIBE

VIBE is another 3D HMR model that predicts the SMPL parameters of human motion given a video. Visit the VIBE github repo (listed below) and follow the instructions in the 'Getting Started' and 'Running the Demo' section of their README.md to obtain the vibe_output.pkl file for your video. This .pkl file should be placed in your .vibe subdirectory (see the next section on directory structure).
[VIBE](https://github.com/mkocabas/VIBE).

## Resulting directory structure

```
/nemo-cvpr2023
-- /data
  | -- /videos
  |   | -- <ACTION>.<INDEX>.mp4
  |   |  ......
  | -- /exps
  |   | -- /your_experiment_directory_<ACTION>
  |   |    | -- /<ACTION>.<INDEX>.frames
  |   |    | -- /<ACTION>.<INDEX>.op
  |   |    | -- /<ACTION>.<INDEX>.vibe
  |   |  ......
  | -- opt_cam_IMG_6287.pt
  | -- opt_cam_IMG_6289.pt

```

## Running NeMo

If you have reached this section, you have completed the video pre-processing step and are ready to run NeMo.

### Prepare a config file (.yml)

First, you must prepare a config file within the 'nemo/config' subdirectory. Check out other pre-existing config files for reference, but below is an example template for your config file.

```
exp_dir: "/your/group/directory/nemo-cvpr2023/data/exps/your_experiment_directory"
format: "release"
videos:
  root_dir: "/your/group/directory/nemo-cvpr2023/data/videos"
  names:
    [
      "tennis_swing.0",
      "tennis_swing.1",
      "tennis_swing.2",
      "tennis_swing.3",
      "tennis_swing.4",
      "tennis_swing.5",
    ]

```

### Prepare a script file

You can either prepare your own .sh file or modify pre-existing files. A simple way to do this would be to edit the 'jeff_scripts/20230427-mv3-gear-jeff.sh' script file. Modify the --nemo_cfg_path so that it points to your .yml file created in the previous section, and the --out_dir to whichever directory you would like the NeMo outputs to be saved to.

Then, run the following command:

```bash
bash jeff_scripts/20230427-mv3-gear-jeff.sh 0
```
