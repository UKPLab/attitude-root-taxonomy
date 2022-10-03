#!/bin/bash
#
#SBATCH --job-name=0_7_roberta_zero
#SBATCH --output=/ukp-storage-1/bates/setfit/final_code/nhb-attitude-root-taxonomy/output_slurm/0_7_robertatzero.txt
#SBATCH --mail-user=bates@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --account=athene-researcher
#SBATCH --partition=athene
#SBATCH --time=02:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

source /storage/ukp/work/bates/setfit/final_code/nhb-attitude-root-taxonomy/mvenv/bin/activate
module purge
module load cuda/11.1
TRANSFORMERS_OFFLINE=1 \
python /storage/ukp/work/bates/setfit/final_code/nhb-attitude-root-taxonomy/main.py --TRANSFORMER_CLF='roberta-base'\
                                                                       --FOLD=0\
                                                                       --ST_MODEL='paraphrase-mpnet-base-v2'\
                                                                       --NUM_ROOTS=7\
                                                                       --MODE='transformer_zero_shot'\
                                                                       --EPOCHS=0\
                                                                       --PRETRAIN_EPOCHS=10\

                                                                       