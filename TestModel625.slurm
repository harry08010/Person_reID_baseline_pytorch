#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --mail-type=ALL
#SBATCH --mail-user=harry08010@tamu.edu
#SBATCH --job-name=TestModel625
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=6144M
#SBATCH --output=TestResult.%j
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


module load PyTorch/0.3.1-foss-2018a-Python-3.6.4-CUDA-9.1.85
module load torchvision/0.2.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module swap Python/3.6.4-golf-2018a Python/3.6.4-foss-2018a
# python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir ../Market-1501-v15.09.15/pytorch/
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ../Market-1501-v15.09.15/ptest/  --batchsize 32 --which_epoch last
python evaluate_gpu.py