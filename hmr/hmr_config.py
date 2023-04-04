"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join
# EFT_ROOT_FOLDER ='/home/users/wangkua1/projects/eft/'
# EFT_DATASET_NPZ_PATH = EFT_ROOT_FOLDER+'preprocessed_db/'
# H36M_ROOT = '/scratch/groups/syyeung/hmr_datasets/h36m_val'
# LSP_ROOT = '/scratch/groups/syyeung/hmr_datasets/lsp'
# LSP_ORIGINAL_ROOT = '/scratch/groups/syyeung/hmr_datasets/lsp_orig'
# LSPET_ROOT = '/scratch/groups/syyeung/hmr_datasets/hr-lspet'
# MPII_ROOT = '/scratch/groups/syyeung/hmr_datasets/mpii_human_pose'
# COCO_ROOT = '/scratch/groups/syyeung/hmr_datasets/coco'
# MPI_INF_3DHP_ROOT = '/scratch/groups/syyeung/hmr_datasets/mpi_inf_3dhp_full'
# PW3D_ROOT = '/scratch/groups/syyeung/hmr_datasets/3dpw'
# UPI_S1H_ROOT = '' # not needed for training.


# TEST_SPLIT=0
# TRAIN_SPLIT=1
# # Path to test/train npz files
# DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
#                    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
#                    'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
#                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
#                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_new.npz'),
#                    '3dpw_val': join(DATASET_NPZ_PATH, '3dpw_validation.npz'),
#                    '3dpw_train': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
#                    '3dpw_train_eft': join(EFT_DATASET_NPZ_PATH, '3dpw_train_subset_False_22663_subjId.npz'), 
#                    '3dpw_test_eft': join(EFT_DATASET_NPZ_PATH, '3dpw_test_34561_subjId.npz'), 
#                    '3dpw_train_2d': join(DATASET_NPZ_PATH, '3dpw_train_hidesmpl.npz'),
#                    'seedlings': join(DATASET_NPZ_PATH, 'seedlings_dataset.npz'),
#                    'seedlings2': join(DATASET_NPZ_PATH, 'seedlings_dataset_new.npz'), # this is the one we tested on
                   
#                    'agora_valid': join(DATASET_NPZ_PATH, 'agora_valid.npz'),
#                    'agora_validation_keypoints': join(DATASET_NPZ_PATH, 'agora_valid_keypoints.npz'),
#                    'agora_test_keypoints': join(DATASET_NPZ_PATH, 'agora_test_keypoints.npz'),
#                    'dance2': join(DATASET_NPZ_PATH, 'cmu_dance2.npz'),
#                    'dance5': join(DATASET_NPZ_PATH, 'cmu_dance5.npz'),
#                    'climber': join(DATASET_NPZ_PATH, 'climber.npz'),
#                    'jazz': join(DATASET_NPZ_PATH, 'rO3R9KWb4fk.npz'),
#                    'gymnastics': join(DATASET_NPZ_PATH, 'PSBOjqCtpEU.npz')
#                   },
#                   {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
#                    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
#                    'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
#                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
#                    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
#                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
#                    'seedlings_train': join(DATASET_NPZ_PATH, 'seedlings_dataset_train.npz'),
#                    'seedlings_train_infant': join(DATASET_NPZ_PATH, 'seedlings_dataset_train_infant.npz'),
#                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_new.npz'),
#                    '3dpw_train': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
# #                    'agora_train': join(DATASET_NPZ_PATH, 'agora_train.npz'),
#                    '3dpw_train_2d': join(DATASET_NPZ_PATH, '3dpw_train_hidesmpl.npz'),
#                    '3dpw_train_eft': join(EFT_DATASET_NPZ_PATH, '3dpw_train_subset_False_22663_subjId.npz'), 
#                    '3dpw_test_eft': join(EFT_DATASET_NPZ_PATH, '3dpw_test_34561_subjId.npz'), 
#                    'agora_train': join(DATASET_NPZ_PATH, 'agora_train_keypoints.npz'),
#                    'dance2': join(DATASET_NPZ_PATH, 'cmu_dance2.npz'),
#                    'dance5': join(DATASET_NPZ_PATH, 'cmu_dance5.npz'),
#                    'gymnastics': join(DATASET_NPZ_PATH, 'gymnastics.npz'),
#                    'jazz': join(DATASET_NPZ_PATH, 'jazz_train.npz'),
#                   }
#                 ]

# DATASET_FOLDERS = {'h36m': '/scratch/users/wangkua1/data/h36m/images',
#                 }

CUBE_PARTS_FILE = 'software/spin_data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'software/spin_data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'software/spin_data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'software/spin_data/vertex_texture.npy'
STATIC_FITS_DIR = 'software/spin_data/static_fits'
SMPL_MEAN_PARAMS = 'software/spin_data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'software/smpl'
