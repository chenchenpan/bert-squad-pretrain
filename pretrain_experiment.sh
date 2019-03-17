BERT_BASE_DIR=$HOME/projects/bert-squad-pretrain/uncased_L-12_H-768_A-12
OUTPUT_DIR=$HOME/projects/bert-squad-pretrain/output_dir


python create_pretraining_data.py \
  --input_file=./pretrain_squad_fake.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

  python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=$OUTPUT_DIR/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5


# python create_pretraining_data.py \
#   --input_file=./sample_text.txt \
#   --output_file=$OUTPUT_DIR/tf_examples.tfrecord \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=128 \
#   --max_predictions_per_seq=20 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5


# python run_pretraining.py \
#   --input_file=$OUTPUT_DIR/tf_examples.tfrecord \
#   --output_dir=$OUTPUT_DIR/pretraining_output \
#   --do_train=True \
#   --do_eval=True \
#   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#   --train_batch_size=32 \
#   --max_seq_length=128 \
#   --max_predictions_per_seq=20 \
#   --num_train_steps=20 \
#   --num_warmup_steps=10 \
#   --learning_rate=2e-5

  #   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \