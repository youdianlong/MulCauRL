from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
#CUDA_VISIBLE_DEVICES=1 python eval.py --k 10 --models_exp_code /data1/zyc/project/causal_lung/rt_monitoring1/task_2_tumor_subtyping_CLAM_12.5_lcp_causal100_s1  --save_exp_code task_2_tumor_subtyping_CLAM_50_s1_cv --task task_2_tumor_subtyping --model_type clam_mb --results_dir results --data_root_dir  /data1/zyc/result_data/Lung/ --embed_dim 1024

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'MulCauRL', 'mil'], default='MulCauRL', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--name', type=str, default='CRA', help='Model name (default: CRA)')
parser.add_argument('--baseline', type=str, default='refine', help='Baseline (default: refine)')
parser.add_argument('--lan', type=str, default='Cli-BERT', help='Language model (default: Cli-BERT)')
parser.add_argument('--lan_weight_path', type=str, default="/data1/zyc/project/causal_lung/dataset_modules/Clinical-BERT/", help='Path to pre-trained language model')
parser.add_argument('--feature_dim', type=int, default=768, help='Feature dimension (default: 768)')
parser.add_argument('--word_dim', type=int, default=768, help='Word dimension (default: 768)')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers (default: 2)')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (default: 8)')
parser.add_argument('--d_model', type=int, default=768, help='Model dimension (default: 768)')
parser.add_argument('--d_ff', type=int, default=768, help='Feed-forward dimension (default: 768)')
parser.add_argument('--vocab_size', type=int, default=50265, help='Vocabulary size (default: 50265)')
parser.add_argument('--n_negs', type=int, default=1, help='Number of negative samples (default: 1)')
parser.add_argument('--prop_num', type=int, default=1, help='Number of proposals (default: 1)')
parser.add_argument('--sigma', type=float, default=9, help='Sigma value (default: 9)')
parser.add_argument('--gamma', type=float, default=1, help='Gamma value (default: 1)')
parser.add_argument('--lamb', type=float, default=0.15, help='Overlap control for proposals (default: 0.15)')
parser.add_argument('--window_sizes', type=int, nargs='+', default=[1, 3, 5], help='Window sizes (default: [1, 3, 5])')
parser.add_argument('--time_penalty', type=float, default=0.1, help='Time penalty (default: 0.1)')
parser.add_argument('--do', action='store_true', default=False, help='Enable causal intervention (default: False)')
parser.add_argument('--do_patch', action='store_true', default=False, help='Enable causal intervention (default: False)')
parser.add_argument('--vg_loss', type=float, default=1, help='Video grounding loss (default: 1)')
parser.add_argument('--vote', type=int, default=0, help='Vote for temporal proposal during inference (default: 0)')
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/data1/zyc/project/causal_lung/results_features/case_slide_ids_Lung.csv',
                            data_dir= os.path.join('/data1/zyc/result_data/', 'Lung/'),
                            shuffle = False,  
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1},
                            patient_strat= False,
                            ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/task_2_tumor_subtyping_100/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(patient_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
