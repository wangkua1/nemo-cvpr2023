#!/bin/bash
export PATH="/pasteur/u/jeffheo/miniconda3/bin:$PATH"

# Name of the yml file without the extension
vibe_name="vibe_environment"
# Create the environment from the yml file
conda env create -f ${vibe_name}.yml
# Activate the environment
conda init bash
conda activate $vibe_name

video_names=$(yq e '.videos.names[]' ./nemo-config.yml)

# change write permissions for the data folder so that the docker container can write to it
chmod 777 ./data

# iterate over video names
for name in $video_names; do
    python ./VIBE_custom/demo.py --vid_file ./data/videos/${name}.mp4 --output_folder ./data/exps/${name}.vibe/ --display
done

# Deactivate the environment
conda deactivate
# create conda environment through custom_env.yml
env_name="custom_env"
conda env create -f ${env_name}.yml
# Activate the environment
conda activate $env_name
#iterate over video names 
#get the number of videos
num_videos=$(yq e '.videos.names | length' ./nemo-config.yml)
count=$((num_videos))
for name in $video_names; do
    #get the name of the video before the . in the name
    video_name=$(echo $name | cut -d'.' -f 1)
    python ./video_to_frames_custom.py --action $video_name --num_videos $count  --display
    docker run --gpus 5 --rm -v /pasteur/u/jeffheo/projects/nemo-cvpr2023-jeff/custom_video/data/exps:/mnt cwaffles/openpose ./build/examples/openpose/openpose.bin --image_dir /mnt/${name}.frames --write_json /mnt/${name}.op --display 0  --model_pose BODY_25 --number_people_max 1 --render_pose 0
done

# Now run the NeMo script
bash ./nemo-run.sh 0
