for lr_human in 1e-4 1e-3; do
for lr_phase in 0; do
for weight_gmm_loss in 1; do
for weight_vp_loss in 10; do
for weight_vp_z_loss in 1; do
for h_dim in 1000; do
for phase_rbf_dim in 100; do
for instance_style in 2; do
for warmup_step in 300; do
for rbf_kernel in 'quadratic'; do
for lr_factor in 1; do
for loss in mse_robust; do
for action in tennis-swing; do
for label_type in op; do
for weight_3d_loss in 1000; do
for weight_humor_loss in 1e-3; do


case $instance_style in
  0)
    instance_code_size=0
    lr_instance=0
    ;;
  1)
    instance_code_size=2
    lr_instance=1e-3
    ;;
2)
    instance_code_size=5
    lr_instance=1e-3
    ;;
esac


n_steps=2000

cur_fname="$(basename $0 .sh)"
expname=${cur_fname}-${action}-${lr_human}-${lr_instance}-${instance_code_size}-${lr_phase}-${weight_gmm_loss}-${weight_vp_loss}-${weight_vp_z_loss}-${h_dim}-${phase_rbf_dim}-${rbf_kernel}-${warmup_step}-${lr_factor}-${loss}-${label_type}-${weight_3d_loss}

partition=pasteur

cmd="python -m scripts.learned_multi_view_recon_nn_jeff \
    --default_config config/default-v1.yml \
    --label_type ${label_type} \
    --data_loader_type demo \
    --nemo_cfg_path nemo/config/nemo_additional_jeff/trimmed/${action}.yml \
    --out_dir shared_out/jeff_out/trimmed_out/${expname} \
    --lr_factor ${lr_factor} \
    --batch_size 512 \
    --n_steps ${n_steps} \
    --warmup_step ${warmup_step} \
    --opt_cam_step 1000 \
    --model_version 3 \
    --phase_rbf_dim ${phase_rbf_dim} \
    --rbf_kernel ${rbf_kernel} \
    --loss ${loss} \
    --lr_phase ${lr_phase} \
    --weight_gmm_loss ${weight_gmm_loss} \
    --weight_vp_loss ${weight_vp_loss} \
    --weight_vp_z_loss ${weight_vp_z_loss} \
    --weight_humor_loss ${weight_humor_loss} \
    --lr_human ${lr_human} \
    --lr_instance ${lr_instance} \
    --instance_code_size ${instance_code_size} \
    --h_dim ${h_dim} \
    --render_rollout_figure \
    --weight_3d_loss ${weight_3d_loss} \
    --render_video 1 \
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
#SBATCH --output=..slurm_logs/jeff/${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=..slurm_logs/jeff/${cur_fname}-${partition}-%j-err.txt
#SBATCH --mem=64gb
#SBATCH --account=${partition}
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

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done