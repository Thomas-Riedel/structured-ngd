#!/bin/bash

python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model TempScaling
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model BBB         --mc_samples_val 8 --mc_samples_test 64
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model LLDropout   --mc_samples_val 8 --mc_samples_test 64
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model Dropout     --mc_samples_val 8 --mc_samples_test 64

# Deep Ensemble with 5 models
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32
python main.py --optimizer SGD --lr 1e-2 --n_bins 10 --epochs 250 --dataset CIFAR10 --model DeepEnsemble

# Run NGD with multiple values for k, M, and gamma for ablation study and HyperDeepEnsemble
python main.py --optimizer StructuredNGD --structure rank_cov  --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k 0,1,3,5 --mc_samples_val 8 --mc_samples_test 64 --gamma 1.0 --mc_samples 1
python main.py --optimizer StructuredNGD --structure arrowhead --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k   1,3,5 --mc_samples_val 8 --mc_samples_test 64 --gamma 1.0 --mc_samples 1

python main.py --optimizer StructuredNGD --structure rank_cov  --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k 0,1,3,5 --mc_samples_val 8 --mc_samples_test 64 --gamma 0.1
python main.py --optimizer StructuredNGD --structure arrowhead --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k   1,3,5 --mc_samples_val 8 --mc_samples_test 64 --gamma 0.1

python main.py --optimizer StructuredNGD --structure rank_cov  --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k 0,1,3,5 --mc_samples_val 8 --mc_samples_test 64 --gamma 0.0
python main.py --optimizer StructuredNGD --structure arrowhead --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k   1,3,5 --mc_samples_val 8 --mc_samples_test 64 --gamma 0.0

python main.py --optimizer StructuredNGD --structure rank_cov  --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k 0,1,3,5 --mc_samples_val 8 --mc_samples_test 64 --mc_samples 3
python main.py --optimizer StructuredNGD --structure arrowhead --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k   1,3,5 --mc_samples_val 8 --mc_samples_test 64 --mc_samples 3

python main.py --optimizer StructuredNGD --structure rank_cov  --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k 0,1,3,5 --mc_samples_val 8 --mc_samples_test 64 --mc_samples 5
python main.py --optimizer StructuredNGD --structure arrowhead --lr 1e-1 --n_bins 10 --epochs 250 --dataset CIFAR10 --model ResNet32 --k   1,3,5 --mc_samples_val 8 --mc_samples_test 64 --mc_samples 5

# Hyper-Deep Ensembles for individual M with gamma set to 1 and all possible values k for structure Rank Covariance
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=rank_cov;M=1'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=rank_cov;M=3'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=rank_cov;M=5'

# Hyper-Deep Ensembles for individual gamma with M set to 1 and all possible values k for structure Rank Covariance
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=rank_cov;gamma=1.0'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=rank_cov;gamma=0.1'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=rank_cov;gamma=0.0'

# Hyper-Deep Ensembles for individual M with gamma set to 1 and all possible values k for structure Arrowhead
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=arrowhead;M=1'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=arrowhead;M=3'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=arrowhead;M=5'

# Hyper-Deep Ensembles for individual gamma with M set to 1 and all possible values k for structure Arrowhead
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=arrowhead;gamma=1.0'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=arrowhead;gamma=0.1'
python main.py ---n_bins 10 --dataset CIFAR10 --mc_samples_val 8 --mc_samples_test 64 --model 'HyperDeepEnsemble|structure=arrowhead;gamma=0.0'