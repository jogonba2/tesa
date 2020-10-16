#!/bin/bash

# Parameters
TASK_TYPE=$1
CONTEXT_FORMAT=$2
TARGETS_FORMAT=$3
SYMBOLIC_INFO=$4
SYMBOLIC_FORMAT=$5 # Format is not required in case of use the symbolic info as input
TASK=context-dependent-same-type
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=24
BATCH_SIZE=4
SOFT_LABELS=false
FURTHER_GEN_FINETUNING=true # If true, start the finetuning from the weights of discriminative BART.
FREEZING_STRATEGY=gradual_unfreezing

BART=bart.large.cnn

# Finetuning parameters
if [ $TASK_TYPE == "classification" ]
then
  MAX_EPOCHS=6
  MAX_TOKENS=4400
  MAX_SENTENCES=2
  UPDATE_FREQ=1
  LR=2e-05
  WARMUP_UPDATES_PERCENT=6
elif [ $TASK_TYPE == "generation" ]
then
  MAX_EPOCHS=6
  MAX_TOKENS=1024
  UPDATE_FREQ=1
  LR=5e-06
  WARMUP_UPDATES_PERCENT=3
fi

EXPERIMENT=ep"$MAX_EPOCHS"_tok"$MAX_TOKENS"_sent"$MAX_SENTENCES"_freq"$UPDATE_FREQ"_lr"$LR"_warm"$WARMUP_UPDATES_PERCENT"

# Paths
MASTER_THESIS_PATH=/home/jogonba2/Escritorio/EstanciaMontreal/tesa_collaboration
PREPROCESSED_DATA_PATH=/home/jogonba2/Escritorio/EstanciaMontreal/tesa_collaboration/results/preprocessed_data
PRETRAINED_MODELS_PATH=/home/jogonba2/Escritorio/EstanciaMontreal/tesa_collaboration/pretrained_models
CHECKPOINTS_PATH=/home/jogonba2/Escritorio/EstanciaMontreal/tesa_collaboration/results/checkpoints
SLURM_TMPDIR=/home/jogonba2/Escritorio/EstanciaMontreal/tesa_collaboration/slurm_dir/

# Recover full paths/names
FULL_TASK="$TASK"_"$TRAIN_PROPORTION"-"$VALID_PROPORTION"-"$TEST_PROPORTION"_rs"$RANKING_SIZE"_bs"$BATCH_SIZE"_cf-"$CONTEXT_FORMAT"_tf-"$TARGETS_FORMAT"

if [ ! -z "${SYMBOLIC_INFO}" ]
then
FULL_TASK="$FULL_TASK"_"sym"-"$SYMBOLIC_INFO"
fi

if [ ! -z "${SYMBOLIC_FORMAT}" ]
then
FULL_TASK="$FULL_TASK"_"symf"-"$SYMBOLIC_FORMAT"
fi

if [ $SOFT_LABELS == true ]
then
FULL_TASK="$FULL_TASK"_"soft"
SOFT_LABELS_MAP='{"not_aggregation":0,"aggregation":1,"partial_aggregation":0.6}';
fi

if [ $FURTHER_GEN_FINETUNING == true ]
then
INITIAL_CHECKPOINT="classification/context-dependent-same-type_50-25-25_rs24_bs4_cf-v0_tf-v0/ep6_tok4400_sent8_freq1_lr2e-05_warm6/checkpoint3.pt"
fi

RESULTS_PATH="$CHECKPOINTS_PATH/$TASK_TYPE/$FULL_TASK/$EXPERIMENT"

# Print the parameters
echo "Parameters:"; echo $TASK_TYPE $CONTEXT_FORMAT $EXPERIMENT; echo
echo "Context format:"; echo $CONTEXT_FORMAT; echo
echo "Target format:"; echo $TARGETS_FORMAT; echo
echo "Symbolic info:"; echo $SYMBOLIC_INFO; echo
echo "Symbolic format:"; echo $SYMBOLIC_FORMAT; echo
echo "Further finetuning:"; echo $FURTHER_GEN_FINETUNING; echo
echo "Initial checkpoint:"; echo $INITIAL_CHECKPOINT; echo
echo "Freezing strategy:"; echo $FREEZING_STRATEGY; echo
echo "Results path:"; echo $RESULTS_PATH; echo

# Load miniconda
module load miniconda
source activate base
source activate nlp

# Load pretrained BART
#tar -xf "$PRETRAINED_MODELS_PATH/$BART.tar.gz" -C $SLURM_TMPDIR

# Discriminative model as initial weights for the generative model.
if [ $FURTHER_GEN_FINETUNING == true ]
then
  rm "$SLURM_TMPDIR/$BART/model.pt"
  cp "$CHECKPOINTS_PATH/$INITIAL_CHECKPOINT" "$SLURM_TMPDIR/$BART/model.pt"
fi

echo "Uncompressed BART"

# Load the preprocessed_data
cp -r "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin" $SLURM_TMPDIR

# Re-initialize the results folder
#rm -rf $RESULTS_PATH
#mkdir -p $RESULTS_PATH/tensorboard_logs

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

if [ $TASK_TYPE == "classification" ]
then
  # Compute the number of updates
  if [ $CONTEXT_FORMAT == "v0" ]
  then
    #NUM_UPDATES_PER_EPOCH=2709  #for max_sentences=16 or above, max_tokens=4400
    NUM_UPDATES_PER_EPOCH=3030  #for max_sentences=8, max_tokens=4400
    #NUM_UPDATES_PER_EPOCH=5148  #for max_sentences=4, max_tokens=4400
  elif [ $CONTEXT_FORMAT == "v1" ]
  then
    NUM_UPDATES_PER_EPOCH=9435
  elif [ $CONTEXT_FORMAT == "v2" ]
  then
    NUM_UPDATES_PER_EPOCH=9493
  elif [ $CONTEXT_FORMAT == "v3" ]
  then
    NUM_UPDATES_PER_EPOCH=9618
  elif [ $CONTEXT_FORMAT == "v4" ]
  then
    NUM_UPDATES_PER_EPOCH=9435
  elif [ $CONTEXT_FORMAT == "va" ]
  then
    NUM_UPDATES_PER_EPOCH=4076
  elif [ $CONTEXT_FORMAT == "vb" ]
  then
    NUM_UPDATES_PER_EPOCH=6690
  elif [ $CONTEXT_FORMAT == "vc" ]
  then
    NUM_UPDATES_PER_EPOCH=2574
  else
    NUM_UPDATES_PER_EPOCH=9365
  fi

  TOTAL_NUM_UPDATES=$(($NUM_UPDATES_PER_EPOCH * $MAX_EPOCHS / $UPDATE_FREQ))
  WARMUP_UPDATES=$(($WARMUP_UPDATES_PERCENT * $TOTAL_NUM_UPDATES / 100))

  # Print the parameters
  echo "Finetuning parameters:"; echo $MAX_EPOCHS; echo $MAX_SENTENCES; echo $UPDATE_FREQ; echo $LR;
  echo $WARMUP_UPDATES_PERCENT; echo $NUM_UPDATES_PER_EPOCH; echo $TOTAL_NUM_UPDATES; echo $WARMUP_UPDATES; echo

  CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$FULL_TASK-bin" \
      --max-epoch $MAX_EPOCHS \
      --max-sentences $MAX_SENTENCES \
      --max-tokens $MAX_TOKENS \
      --update-freq $UPDATE_FREQ \
      --lr-scheduler polynomial_decay \
      --lr $LR \
      --total-num-update $TOTAL_NUM_UPDATES \
      --warmup-updates $WARMUP_UPDATES \
      --restore-file $BART/model.pt \
      --save-dir $RESULTS_PATH \
      --tensorboard-logdir $RESULTS_PATH/tensorboard_logs \
      --task sentence_prediction \
      --add-prev-output-tokens \
      --layernorm-embedding \
      --share-all-embeddings \
      --share-decoder-input-output-embed \
      --reset-optimizer \
      --reset-dataloader \
      --reset-meters \
      --required-batch-size-multiple 1 \
      --init-token 0 \
      --arch bart_large \
      --criterion sentence_prediction \
      --num-classes 2 \
      --dropout 0.1 \
      --attention-dropout 0.1 \
      --weight-decay 0.01 \
      --optimizer adam \
      --adam-betas "(0.9, 0.98)" \
      --adam-eps 1e-08 \
      --clip-norm 0.0 \
      --best-checkpoint-metric accuracy \
      --maximize-best-checkpoint-metric \
      --no-last-checkpoints \
      --skip-invalid-size-inputs-valid-test \
      --find-unused-parameters \
      `if [ ! -z $SOFT_LABELS_MAP ]; then echo "--soft-labels $SOFT_LABELS_MAP"; fi`;

elif [ $TASK_TYPE == "generation" ]
then
  # Compute the number of updates
  if [ $CONTEXT_FORMAT == "v0" ]
  then
    if [ $TARGETS_FORMAT == "v2" ] || [ $TARGETS_FORMAT == "v3" ]
    then
      if [ ! -z "${SYMBOLIC_INFO}" ]
      then
        if [ -z "${SYMBOLIC_FORMAT}" ] || [ "$SYMBOLIC_FORMAT" == "input" ]
        then
          #NUM_UPDATES_PER_EPOCH=410 # for max_tokens=2048 and --tf v2 with LCA (k-6)
          #NUM_UPDATES_PER_EPOCH=909 # for max_tokens=1024 and --tf v2 wih LCA (k-6)
          #NUM_UPDATES_PER_EPOCH=1427 # for max_tokens=1024 and --tf v2 with IVI hops-2 k-100
          #NUM_UPDATES_PER_EPOCH=918 # for max_tokens=1024 and --tf v2 with IVI hops-2 k-10
          #NUM_UPDATES_PER_EPOCH=857 # for max_tokens=1024 and --tf v2 with IVI hops-1 k-all
          #NUM_UPDATES_PER_EPOCH=937 # for max_tokens=1024 and --tf v2 with IVI hops-1 k-all and LCA (k-6)
          NUM_UPDATES_PER_EPOCH=1002 # for max_tokens=1024 and --tf v2 with IVI hops-2 k-10 and LCA (k-6)
        else
          #NUM_UPDATES_PER_EPOCH=2317 # for max_tokens=1024 and --tf v2 with LCA (k-6) as positive candidates.
          #NUM_UPDATES_PER_EPOCH=1258 # for max_tokens=1024 and --tf v2 with IVI hops-1 k-all as positive candidates.
          NUM_UPDATES_PER_EPOCH=2592 # for max_tokens=1024 and --tf v2 with LCA (k-6) + IVI (hops-1 k-all) as positives.
        fi
      else
        #NUM_UPDATES_PER_EPOCH=381 # for max_tokens=2048 and --tf v2 without LCA
        NUM_UPDATES_PER_EPOCH=835 # for max_tokens=1024 and --tf v2 without LCA
      fi
    else
      NUM_UPDATES_PER_EPOCH=829  # for max_tokens=1024, no max_sentences
      #NUM_UPDATES_PER_EPOCH=375  # for max_tokens=2048, no max_sentences
    fi
  elif [ $CONTEXT_FORMAT == "v1" ]
  then
    NUM_UPDATES_PER_EPOCH=838
  elif [ $CONTEXT_FORMAT == "v2" ]
  then
    NUM_UPDATES_PER_EPOCH=852
  elif [ $CONTEXT_FORMAT == "v3" ]
  then
    NUM_UPDATES_PER_EPOCH=866
  elif [ $CONTEXT_FORMAT == "v4" ]
  then
    NUM_UPDATES_PER_EPOCH=838
  elif [ $CONTEXT_FORMAT == "va" ]
  then
    NUM_UPDATES_PER_EPOCH=282
  elif [ $CONTEXT_FORMAT == "vb" ]
  then
    NUM_UPDATES_PER_EPOCH=517
  elif [ $CONTEXT_FORMAT == "vc" ]
  then
    NUM_UPDATES_PER_EPOCH=24  # for no max-sentences
    #NUM_UPDATES_PER_EPOCH=144  # for max-sentences=16
  elif [ $CONTEXT_FORMAT == "v_ace" ]
  then
    NUM_UPDATES_PER_EPOCH=171 # for max_tokens=2048, no max_sentences and tf-v2
  elif [ $CONTEXT_FORMAT == "v_ae" ]
  then
    NUM_UPDATES_PER_EPOCH=46 # for max_tokens=2048, no max_sentences and tf-v2
  elif [ $CONTEXT_FORMAT == "v_a_or_b_ce" ]
  then
    NUM_UPDATES_PER_EPOCH=210 # for max_tokens=2048, no max_sentences and tf-v2
  elif [ $CONTEXT_FORMAT == "v_blocks1" ]
  then
    NUM_UPDATES_PER_EPOCH=968 # for max_tokens=1024, no max_sentences, tf-v2, LCA k=6, IVI 1-hops k=all
  elif [ $CONTEXT_FORMAT == "v_blocks2" ]
  then
    NUM_UPDATES_PER_EPOCH=968 # for max_tokens=1024, no max_sentences, tf-v2, LCA k=6, IVI 1-hops k=all
  else
    NUM_UPDATES_PER_EPOCH=831
  fi

  # Compute the number of updates
  TOTAL_NUM_UPDATES=$(($NUM_UPDATES_PER_EPOCH * $MAX_EPOCHS / $UPDATE_FREQ))
  WARMUP_UPDATES=$(($WARMUP_UPDATES_PERCENT * $TOTAL_NUM_UPDATES / 100))

  # Print the parameters
  echo "Finetuning parameters:"; echo $MAX_EPOCHS; echo $MAX_TOKENS; echo $UPDATE_FREQ; echo $LR;
  echo $WARMUP_UPDATES_PERCENT; echo $NUM_UPDATES_PER_EPOCH; echo $TOTAL_NUM_UPDATES; echo $WARMUP_UPDATES; echo

  # Run the finetuning
  echo "Training on $FULL_TASK-bin"; echo
  CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$FULL_TASK-bin" \
      --max-epoch $MAX_EPOCHS \
      --max-tokens $MAX_TOKENS \
      --update-freq $UPDATE_FREQ \
      --lr-scheduler polynomial_decay \
      --lr $LR \
      --total-num-update $TOTAL_NUM_UPDATES \
      --warmup-updates $WARMUP_UPDATES \
      --restore-file $BART/model.pt \
      --save-dir $RESULTS_PATH \
      --tensorboard-logdir $RESULTS_PATH/tensorboard_logs \
      --task translation \
      --source-lang source \
      --target-lang target \
      --truncate-source \
      --layernorm-embedding \
      --share-all-embeddings \
      --share-decoder-input-output-embed \
      --reset-optimizer \
      --reset-dataloader \
      --reset-meters \
      --required-batch-size-multiple 1 \
      --arch bart_large \
      --criterion label_smoothed_cross_entropy \
      --label-smoothing 0.1 \
      --dropout 0.1 \
      --attention-dropout 0.1 \
      --weight-decay 0.01 \
      --optimizer adam \
      --adam-betas "(0.9, 0.999)" \
      --adam-eps 1e-08 \
      --clip-norm 0.1 \
      --no-last-checkpoints \
      --freezing-strategy $FREEZING_STRATEGY \
      --find-unused-parameters;

fi
# OJO CON FREEZE ENCODER PARAMS! #

# Remove checkpoint_best.pt
#rm $RESULTS_PATH/checkpoint_best.pt

echo "Done."; echo
