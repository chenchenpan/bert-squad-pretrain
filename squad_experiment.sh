BERT_LARGE_DIR=/Users/cicipan/projects/CS224N/project/bert/uncased_L-12_H-768_A-12
SQUAD_DIR=/Users/cicipan/projects/CS224N/project/bert/squad
OUTPUT_DIR=/Users/cicipan/projects/CS224N/project/output_dir/fake

python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config_fake.json \
  --do_train=True \
  --train_file=$SQUAD_DIR/fake_train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/fake_train-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-2 \
  --num_train_epochs=100.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR\
  --version_2_with_negative=True


python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/fake_train-v2.0.json $OUTPUT_DIR/predictions.json \
  --na-prob-file $OUTPUT_DIR/null_odds.json




    # --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
    #  --learning_rate=3e-5 \
