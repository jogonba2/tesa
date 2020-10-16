#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --gres=gpu:32gb:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=6:00:00
#SBATCH --error=/network/tmp1/jose-angel.gonzalez-barba/logs/rank-%j.err
#SBATCH --output=/network/tmp1/jose-angel.gonzalez-barba/logs/rank-%j.out

# Parameters
RANKER=$1
CONTEXT_FORMAT=$2
TARGETS_FORMAT=$3

if [ $RANKER == "discriminative" ]
then
  CONTEXT_FORMAT="v0"
  TARGETS_FORMAT="v0"
else
  CONTEXT_FORMAT="v0"
  TARGETS_FORMAT="v2"
fi

TASK=context-dependent-same-type
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=24
BATCH_SIZE=4
SOFT_LABELS=false
BART=bart.large.cnn

# Paths
MASTER_THESIS_PATH=/home/mila/j/jose-angel.gonzalez-barba/tesa_collaboration
PRETRAINED_MODELS_PATH=/network/tmp1/jose-angel.gonzalez-barba/pretrained_models
PREPROCESSED_DATA_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/preprocessed_data
TASKS_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/modeling_task
CHECKPOINTS_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/checkpoints

# Recover full paths/names
FULL_TASK=context-dependent-same-type_50-25-25_rs24_bs4_cf-v0_tf-v0
GENERATIVE_PATH=$CHECKPOINTS_PATH/generation/context-dependent-same-type_50-25-25_rs24_bs4_cf-v0_tf-v2/ep6_tok1024_sent_freq1_lr5e-06_warm3/checkpoint5.pt
DISCRIMINATIVE_PATH=$CHECKPOINTS_PATH/classification/context-dependent-same-type_50-25-25_rs24_bs4_cf-v0_tf-v0/ep6_tok4400_sent8_freq1_lr2e-05_warm6/checkpoint3.pt

echo "Ranker:"; echo $RANKER; echo
echo "(Ranker) Context format:"; echo $CONTEXT_FORMAT; echo
echo "(Ranker) Target format:"; echo $TARGETS_FORMAT; echo
echo "Task:"; echo "$TASKS_PATH/$FULL_TASK.pkl"; echo

# Load miniconda
module load miniconda
source activate base
source activate nlp

# Load pretrained BART
tar -xf "$PRETRAINED_MODELS_PATH/$BART.tar.gz" -C $SLURM_TMPDIR

# Load the task
cp "$TASKS_PATH/$FULL_TASK.pkl" $SLURM_TMPDIR

# Load the preprocessed data if necessary
if [ $RANKER == "discriminative" ]
then
  cp -r "$PREPROCESSED_DATA_PATH/classification/$FULL_TASK-bin/input0" \
        "$PREPROCESSED_DATA_PATH/classification/$FULL_TASK-bin/input1" \
        "$PREPROCESSED_DATA_PATH/classification/$FULL_TASK-bin/label" \
        $SLURM_TMPDIR/$BART
fi

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

# Makes sure we don't compute *.pt if there is no checkpoint file
shopt -s nullglob

if [ $RANKER == "discriminative" ]
then
  cp $GENERATIVE_PATH $BART
  cp $DISCRIMINATIVE_PATH $BART
  GEN_CHECKPOINT=${GENERATIVE_PATH##*/}
  DISC_CHECKPOINT=${DISCRIMINATIVE_PATH##*/}

  python -u $MASTER_THESIS_PATH/run_model.py \
      --context_format $CONTEXT_FORMAT \
      --targets_format $TARGETS_FORMAT \
      --task_path "$TASKS_PATH/$FULL_TASK.pkl" \
      --ranker discriminative \
      --bart \
      --wild \
      --pretrained_path $BART \
      --checkpoint_generative $GEN_CHECKPOINT \
      --checkpoint_discriminative $DISC_CHECKPOINT \
      --rerank;


elif [ $RANKER == "generative" ]
then
  cp $GENERATIVE_PATH $BART
  HALF_CHECKPOINT=${GENERATIVE_PATH%.*}
  CHECKPOINT=${GENERATIVE_PATH##*/}

  python -u $MASTER_THESIS_PATH/run_model.py \
      --context_format $CONTEXT_FORMAT \
      --targets_format $TARGETS_FORMAT \
      --task_path "$TASKS_PATH/$FULL_TASK.pkl" \
      --ranker generative \
      --bart \
      --wild \
      --pretrained_path $BART \
      --checkpoint_generative $CHECKPOINT \
      --rerank;
fi

echo "Done."; echo
