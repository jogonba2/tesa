#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --time=1:00:00
#SBATCH --error=/network/tmp1/jose-angel.gonzalez-barba/logs/preprocess-%j.err
#SBATCH --output=/network/tmp1/jose-angel.gonzalez-barba/logs/preprocess-%j.out

# Parameters
TASK_TYPE=$1
CONTEXT_FORMAT=$2
TARGETS_FORMAT=$3
SYMBOLIC_INFO=$4
SYMBOLIC_FORMAT=$5 # Format is not required in case of use the symbolic info as input
TASK=extreme-ranking-tesa
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=235
BATCH_SIZE=1
SOFT_LABELS=false

# Paths
MASTER_THESIS_PATH=/home/mila/j/jose-angel.gonzalez-barba/tesa_collaboration
FINETUNING_DATA_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/finetuning_data
PREPROCESSED_DATA_PATH=/network/tmp1/jose-angel.gonzalez-barba/results/preprocessed_data

# Recover full paths/names
FULL_TASK="$TASK"_"$TRAIN_PROPORTION"-"$VALID_PROPORTION"-"$TEST_PROPORTION"_rs"$RANKING_SIZE"_bs"$BATCH_SIZE"_cf-"$CONTEXT_FORMAT"_tf-"$TARGETS_FORMAT"

if [ ! -z "${SYMBOLIC_INFO}" ]
then
FULL_TASK="$FULL_TASK"_"sym"-"$SYMBOLIC_INFO"
fi

if [ ! -z "${SYMBOLIC_FORMAT}" ] &&  [ "$SYMBOLIC_FORMAT" != "input" ]
then
FULL_TASK="$FULL_TASK"_"symf"-"$SYMBOLIC_FORMAT"
fi

if [ $SOFT_LABELS == true ]
then
FULL_TASK="$FULL_TASK"_"soft"
fi

# Print the parameters
echo "Parameters:"; echo $TASK_TYPE; echo
echo "Context format:"; echo $CONTEXT_FORMAT; echo
echo "Target format:"; echo $TARGETS_FORMAT; echo
echo "Symbolic info:"; echo $SYMBOLIC_INFO; echo
echo "Symbolic format:"; echo $SYMBOLIC_FORMAT; echo
echo "Soft labels:"; echo $SOFT_LABELS; echo
echo "Results path:"; echo "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin"; echo

# Load miniconda
module load miniconda
source activate base
source activate nlp

# Load the finetuning data
cp -r "$FINETUNING_DATA_PATH/$TASK_TYPE/$FULL_TASK" $SLURM_TMPDIR

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

if [ $TASK_TYPE == "classification" ]
then
  # Rename the input
  mv $FULL_TASK RTE

  # Run the preprocessing
  $MASTER_THESIS_PATH/fairseq/examples/roberta/preprocess_GLUE_tasks.sh . RTE

  # Rename the output
  mv RTE-bin "$FULL_TASK-bin"

elif [ $TASK_TYPE == "generation" ]
then
  # Load the encoding files
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

  # Run the preprocessing
  for SPLIT in train val
  do
    for LANG in source target
    do
      python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$FULL_TASK/$SPLIT.$LANG" \
        --outputs "$FULL_TASK/$SPLIT.bpe.$LANG" \
        --workers 60 \
        --keep-empty;
    done
  done

  fairseq-preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref "$FULL_TASK/train.bpe" \
      --validpref "$FULL_TASK/val.bpe" \
      --destdir "$FULL_TASK-bin" \
      --workers 60 \
      --srcdict dict.txt \
      --tgtdict dict.txt;
fi

# Re-initialize the results folder
rm -rf "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin"
mkdir -p "$PREPROCESSED_DATA_PATH/$TASK_TYPE"

# Move the data to the server
mv "$FULL_TASK-bin" "$PREPROCESSED_DATA_PATH/$TASK_TYPE"

echo "Done."; echo
