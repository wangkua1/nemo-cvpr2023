action=baseball-pitch
#for action in tennis-serve tennis-swing baseball-swing baseball-pitch golf-swing; do
lr_human=1e-4
lr_phase=1e-4
weight_gmm_loss=1
weight_vp_loss=10
weight_vp_z_loss=1
h_dim=1000
phase_rbf_dim=100
warmup_step=300
rbf_kernel='quadratic'
lr_factor=1
loss=mse_robust
label_type=op
instance_code_size=5
lr_instance=1e-3
n_steps=2000

cur_fname="$(basename $0 .sh)"
expname=${cur_fname}
partition=gpu

cmd="python -m scripts.learned_multi_view_recon_nn \
    --default_config config/default-v1.yml \
    --label_type ${label_type} \
    --data_loader_type generic \
    --nemo_cfg_path nemo/config/mymocap-${action}.yml \
    --out_dir shared_out/${expname} \
    --lr_factor ${lr_factor} \
    --batch_size 512 \
    --n_steps ${n_steps} \
    --warmup_step ${warmup_step} \
    --opt_cam_step 1000 \
    --model_version 2 \
    --phase_rbf_dim ${phase_rbf_dim} \
    --rbf_kernel ${rbf_kernel} \
    --loss ${loss} \
    --lr_phase ${lr_phase} \
    --weight_gmm_loss ${weight_gmm_loss} \
    --weight_vp_loss ${weight_vp_loss} \
    --weight_vp_z_loss ${weight_vp_z_loss} \
    --lr_human ${lr_human} \
    --lr_instance ${lr_instance} \
    --instance_code_size ${instance_code_size} \
    --h_dim ${h_dim} \
    --db
"

if [ $1 == 0 ] 
then
echo $cmd
eval $cmd
break 100
else
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${cur_fname}-${partition}
#SBATCH --output=slurm_logs/${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/${cur_fname}-${partition}-%j-err.txt
#SBATCH --mem=32gb
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p ${partition}
#SBATCH --time=48:00:00

#necessary env
# source /home/users/wangkua1/setup_rl.sh
echo \"$cmd\"
eval \"$cmd\"
"

fi

