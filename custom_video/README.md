# Running NeMo [CVPR2023 Highlight] On Custom Video Data-set

This document will guide you through the procedure of running NeMo on your own multi-view action video dataset.

## NeMo Additional Installation

All detailed instructions on configuring the conda environment as well as additional software downloads are specified in the README.md file. However, the conda environment configuration will be taken care of by our demo.sh script which we have provided detailed instructions below. Hence, simply take care of creating the 'software' directory for now.

## VIBE_custom Additional Installation

Navigate to VIBE_custom. Run the following command:

```bash
source scripts/prepare_data.sh
```

In case the script is buggy, you could alternatively install the softwares manually through a simple procedure. Create a directory titled 'data'. Download the vibe_data.zip file through the following link: https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r . Lastly, place the vibe_data directory within the aforementioned 'VIBE_custom/data/' directory.

## Description

To run NeMo, we first need to pre-process your custom video file to prepare three directories.

1. Individual frames of the video file
2. OpenPose 2D ground-truth labels on each frame
3. VIBE 3D HMR initial results for each frame

Our 'custom_video/demo.sh' shell script handles all these sub-tasks and runs the NeMo algorithm, ultimately storing the output renderings in 'custom_video/out/' directory.

Before running the script, make a directory called 'videos' and 'exps' within the 'custom_video/data/' directory. Transfer your video files into the 'custom_video/data/videos/' directory. Format the filenames of the .mp4 video files as: action.index.mp4 (e.g. tennis_swing.0.mp4).

## Resulting directory structure

```
/nemo-cvpr2023
-- /custom_video
  | -- /data
      | -- /videos
      |   | -- <ACTION>.<INDEX>.mp4
      |   |  ......
      | -- /exps
      | -- opt_cam_IMG_6287.pt
      | -- opt_cam_IMG_6289.pt
```

### Prepare a config file (.yml)

Next, modify the nemo-config.yml file accordingly to your video files.

```
exp_dir: "./custom_video/data/exps/"
format: "release"
videos:
  root_dir: "./custom_video/data/videos/"
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

## Running NeMo

Navigate into the 'custom_video' directory and run:

```bash
bash demo.sh
```
