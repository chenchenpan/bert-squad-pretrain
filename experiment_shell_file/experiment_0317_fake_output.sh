INPUT_DIR=$HOME/projects/bert-squad-pretrain
BERT_BASE_DIR=$HOME/projects/bert-squad-pretrain/uncased_L-12_H-768_A-12
SQUAD_DIR=$HOME/projects/bert-squad-pretrain/squad
OUTPUT_DIR=$HOME/projects/bert-squad-pretrain/squad/output
MAX_SEQ_LEN=128
BATCH_SIZE=32


python create_pretraining_data.py \
  --input_file=$INPUT_DIR/pretrain_squad_fake.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=$MAX_SEQ_LEN \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=$BERT_BASE_DIR/pretrain_on_squad \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=$BATCH_SIZE \
  --max_seq_length=$MAX_SEQ_LEN \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/pretrain_on_squad/model.ckpt-20 \
  --do_train=True \
  --train_file=$SQUAD_DIR/fake_train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/fake_train-v2.0.json \
  --train_batch_size=$BATCH_SIZE \
  --learning_rate=3e-2 \
  --num_train_epochs=2.0 \
  --max_seq_length=$MAX_SEQ_LEN \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR\
  --version_2_with_negative=True

python $SQUAD_DIR/evaluate-v2.0.py \
  $SQUAD_DIR/fake_train-v2.0.json \
  $OUTPUT_DIR/predictions.json \
  --na-prob-file $OUTPUT_DIR/null_odds.json

