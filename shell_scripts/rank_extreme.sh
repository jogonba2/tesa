#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --gres=gpu:32gb:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00
#SBATCH --error=/network/tmp1/jose-angel.gonzalez-barba/logs/rank-%j.err
#SBATCH --output=/network/tmp1/jose-angel.gonzalez-barba/logs/rank-%j.out

# Parameters
TASK_TYPE=$1
CONTEXT_FORMAT=$2
TARGETS_FORMAT=$3
EXPERIMENT=$4
SYMBOLIC_INFO=$5
SYMBOLIC_FORMAT=$6

TASK=extreme-ranking-tesa
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=235
BATCH_SIZE=1
SOFT_LABELS=false
BART=bart.large.cnn

# Paths
MASTER_THESIS_PATH=/home/mila/j/jose-angel.gonzalez-barba/tesa_collaboration
PRETRAINED_MODELS_PATH=/network/tmp1/jose-angel.gonzalez-barba/pretrained_models
PREPROCESSED_DATA_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/preprocessed_data
TASKS_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/modeling_task
CHECKPOINTS_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/checkpoints

# Recover full paths/names
FULL_TASK="$TASK"_"$TRAIN_PROPORTION"-"$VALID_PROPORTION"-"$TEST_PROPORTION"_rs"$RANKING_SIZE"_bs"$BATCH_SIZE"_cf-"$CONTEXT_FORMAT"_tf-"$TARGETS_FORMAT"

if [ ! -z "${SYMBOLIC_INFO}" ]
then
FULL_TASK="$FULL_TASK"_"sym"-"$SYMBOLIC_INFO"
fi

if [ ! -z "${SYMBOLIC_FORMAT}" ] && [ "$SYMBOLIC_FORMAT" != "input" ]
then
FULL_TASK="$FULL_TASK"_"symf"-"$SYMBOLIC_FORMAT"
fi

if [ $SOFT_LABELS == true ]
then
FULL_TASK="$FULL_TASK"_"soft"
fi

RESULTS_PATH="$CHECKPOINTS_PATH/$TASK_TYPE/$FULL_TASK/$EXPERIMENT"

# Print the parameters
echo "Parameters:"; echo $TASK_TYPE $CONTEXT_FORMAT $EXPERIMENT; echo
echo "Context format:"; echo $CONTEXT_FORMAT; echo
echo "Target format:"; echo $TARGETS_FORMAT; echo
echo "Symbolic info:"; echo $SYMBOLIC_INFO; echo
echo "Symbolic format:"; echo $SYMBOLIC_FORMAT; echo
echo "Soft labels:"; echo $SOFT_LABELS; echo
echo "Results path:"; echo $RESULTS_PATH; echo
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
if [ $TASK_TYPE == "classification" ]
then
  cp -r "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin/input0" \
        "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin/input1" \
        "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin/label" \
        $SLURM_TMPDIR/$BART
fi

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

# Makes sure we don't compute *.pt if there is no checkpoint file
shopt -s nullglob

CHECKPOINTS_DIR="/network/tmp1/jose-angel.gonzalez-barba/results/checkpoints/classification/context-dependent-same-type_50-25-25_rs24_bs4_cf-v0_tf-v0/ep6_tok4400_sent8_freq1_lr2e-05_warm6/"

for FULL_CHECKPOINT in $CHECKPOINTS_DIR/*.pt
do
  # Load the checkpoint
  cp $FULL_CHECKPOINT $BART

  # Recover the name of the checkpoint
  HALF_CHECKPOINT=${FULL_CHECKPOINT%.*}
  CHECKPOINT=${HALF_CHECKPOINT##*/}

  # Print the checkpoint
  echo; echo "Evaluating $CHECKPOINT"

  # Create path for the results (error analysis) #
  mkdir -p $RESULTS_PATH/$CHECKPOINT

  if [ $TASK_TYPE == "classification" ]
  then
    # Run the ranking
    python -u $MASTER_THESIS_PATH/run_model.py \
          --task $TASK \
          --batch_size $BATCH_SIZE \
          --ranking_size $RANKING_SIZE \
          --context_format $CONTEXT_FORMAT \
          --targets_format $TARGETS_FORMAT \
          --task_path "" \
          --model classifier-bart \
          --bart \
          --pretrained_path $BART \
          --checkpoint_file $CHECKPOINT.pt \
          `if [ ! -z $SYMBOLIC_INFO ]; then echo "--symbolic_algo $SYMBOLIC_INFO"; fi` \
          `if [ ! -z $SYMBOLIC_FORMAT ] && [ "$SYMBOLIC_FORMAT" != "input" ];
                then echo "--symbolic_format $SYMBOLIC_FORMAT"; fi` \
          `if [ $SOFT_LABELS == true ]; then echo "--soft_labels"; fi` \
          --results_path $RESULTS_PATH/$CHECKPOINT;

  elif [ $TASK_TYPE == "generation" ]
  then
    # Run the ranking
    python -u $MASTER_THESIS_PATH/run_model.py \
          -t $TASK \
          --context_format $CONTEXT_FORMAT \
          --targets_format $TARGETS_FORMAT \
          --task_path "" \
          --model generator-bart \
          --bart \
          --pretrained_path $BART \
          --checkpoint_file $CHECKPOINT.pt \
          `if [ ! -z $SYMBOLIC_INFO ]; then echo "--symbolic_algo $SYMBOLIC_INFO"; fi` \
          `if [ ! -z $SYMBOLIC_FORMAT ] && [ "$SYMBOLIC_FORMAT" != "input" ];
                then echo "--symbolic_format $SYMBOLIC_FORMAT"; fi` \
          --results_path $RESULTS_PATH/$CHECKPOINT;
  fi
done

echo "Done."; echo
