export CUDA_VISIBLE_DEVICES=0
INPUT_DIR=$HOME/projects/bert-squad-pretrain
BERT_BASE_DIR=$HOME/projects/bert-squad-pretrain/uncased_L-12_H-768_A-12

OUTPUT_DIR_1=$HOME/projects/bert-squad-pretrain/output_dir/pretrain_on_squad-train_90k-ftseq_128-0318/
INIT_CKPT_1=$HOME/projects/bert-squad-pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt

OUTPUT_DIR_2=$HOME/projects/bert-squad-pretrain/output_dir/pretrain_on_squad-train_90k-ftseq_128-0318-2dseq_384/
INIT_CKPT_2=$HOME/projects/bert-squad-pretrain/output_dir/pretrain_on_squad-train_90k-ftseq_128-0318/model.ckpt-200

INIT_CKPT_3=$HOME/projects/bert-squad-pretrain/output_dir/pretrain_on_squad-train_90k-ftseq_128-0318-2dseq_384/model.ckpt-100

SQUAD_DIR=$HOME/projects/bert-squad-pretrain/squad
OUTPUT_DIR=$HOME/projects/bert-squad-pretrain/pred-0318-train_90k-ftseq_128-100k-2dseq_384/

MAX_SEQ_LEN_1=128
MAX_PREDS_1=20
BATCH_SIZE_1=12
STEPS_1=200
WARM_UP_1=10

MAX_SEQ_LEN_2=384
MAX_PREDS_2=60
BATCH_SIZE_2=6
STEPS_2=100
WARM_UP_2=10


python create_pretraining_data.py \
  --input_file=$INPUT_DIR/pretrain_squad_fake.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=$MAX_SEQ_LEN_1 \
  --max_predictions_per_seq=$MAX_PREDS_1 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=$OUTPUT_DIR_1 \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT_CKPT_1 \
  --train_batch_size=$BATCH_SIZE_1 \
  --max_seq_length=$MAX_SEQ_LEN_1 \
  --max_predictions_per_seq=$MAX_PREDS_1 \
  --num_train_steps=$STEPS_1 \
  --num_warmup_steps=$WARM_UP_1 \
  --learning_rate=2e-5

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=$OUTPUT_DIR_2 \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT_CKPT_2 \
  --train_batch_size=$BATCH_SIZE_2 \
  --max_seq_length=$MAX_SEQ_LEN_2 \
  --max_predictions_per_seq=$MAX_PREDS_2 \
  --num_train_steps=$STEPS_2 \
  --num_warmup_steps=$WARM_UP_2 \
  --learning_rate=2e-5

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT_CKPT_3 \
  --do_train=True \
  --train_file=$SQUAD_DIR/fake_train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/fake_train-v2.0.json \
  --train_batch_size=$BATCH_SIZE_2 \
  --learning_rate=3e-2 \
  --num_train_epochs=2.0 \
  --max_seq_length=$MAX_SEQ_LEN_2 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR\
  --version_2_with_negative=True

python $SQUAD_DIR/evaluate-v2.0.py \
  $SQUAD_DIR/fake_train-v2.0.json \
  $OUTPUT_DIR/predictions.json \
  --na-prob-file $OUTPUT_DIR/null_odds.json

