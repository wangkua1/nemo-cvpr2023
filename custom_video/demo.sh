#!/bin/bash
set -e  # Exit on any error
export PATH="$PATH:."


# Create the environment from the yml file
source ./VIBE_custom/scripts/install_conda.sh


# Obtain video names from nemo-config.yml
 video_names=($(awk '/names:/{flag=1;next}/]/{flag=0}flag' ./nemo-config.yml | grep -o '"[^"]*"' | sed 's/"//g'))
# # Print the video names one per line
#printf '%s\n' "${video_names[@]}"

# # Change write permissions for the data folder so that the docker container can write to it
script_dir=$(cd "$(dirname "$0")" && pwd)
data_dir="${script_dir}/data"
chmod -R 777 "$data_dir"

cd ./VIBE_custom
# Iterate over video names
for name in "${video_names[@]}"; do
     python ./demo.py --vid_file "../${data_dir}/videos/${name}.mp4" --output_folder "../${data_dir}/exps/"
done

cd ..

# Deactivate the environment
eval "$(conda shell.bash hook)"

conda deactivate

# Create conda environment through custom_env.yml
env_name="nemo-env"
conda env create -f "${env_name}.yml"

eval "$(conda shell.bash hook)"
# Activate the environment
conda activate "$env_name"

pip install --force-reinstall pyopengl==3.1.5

# Get the number of videos
num_elements=${#video_names[@]}
count="$num_elements"
video_name_with_extension=${video_names[0]}

# Remove the extension and the number
video_name="${video_name_with_extension%%.*}"
#echo "$video_name"
#echo "$count"

python ./video_to_frames_custom.py --action "$video_name" --num_videos "$count"
# Iterate over video names
for name in "${video_names[@]}"; do
    docker run --gpus 1 --rm -v "${data_dir}/exps:/mnt" cwaffles/openpose ./build/examples/openpose/openpose.bin --image_dir /mnt/"${name%.mp4}.frames" --write_json /mnt/"${name%.mp4}.op" --display 0 --model_pose BODY_25 --number_people_max 1 --render_pose 0
done
cd ..
# Now run the NeMo script
bash custom_video/nemo-run.sh 0
