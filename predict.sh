export PYTHONPATH=$PYTHONPATH:/home/work/qabot/projects/tmp/entity-linking/src
export WORK_PATH=/home/work/qabot/projects/tmp/entity-linking/src/towers
export BERT_BASE_DIR=/home/work/qabot/cased_L-12_H-768_A-12
gpu_id=$1
CUDA_VISIBLE_DEVICES=${gpu_id} python main2.py \
  --data_dir=$WORK_PATH/AIDA_data/top10 \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --eval_steps=500 \
  --learning_rate=2e-5 \
  --do_lower_case=True \
  --warmup_proportion=0.1 \
  --keep_num_checkpoints=3 \
  --model_dir=model/aida_model_ckp_top10 \
  --output_dir=output/aida_output_data_top10 \
  --save_checkpoints_steps=500 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --every_n_iter=100 \
  --num_train_epochs=5 \
  --start_delay_secs=30 \
  --throttle_secs=600 \
  --do_train=False \
  --do_predict=True \
  --do_write=False \
  --query_max_seq_length=64 \
  --desc_max_seq_length=64 \
  --pooling_mode_mean_tokens=False \
  --pooling_mode_cls_token=True \
  --feature_linkcount_p=True \
  --feature_linkcount=False \
  --feature_coherence=True \
  --feature_topic=True \
  --feature_offset=True \
  --feature_end=True \
  --feature_pv=false \
  --metric_type=f1 \
  --init_checkpoint=model/aida_model_ckp_top10/best_exporter_checkpoint/1659705072274/model.ckpt-4000
