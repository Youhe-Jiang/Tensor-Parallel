# MEGATRON_ARGS is only used to initialize megatron. We don't use the model config params in MEGATRON_ARGS.
# num_layers, hidden_size, num_attention_heads, max_position_embeddings will be overwritten by args below.
MEGATRON_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --max-position-embeddings 512 \
           --micro-batch-size 1 \
           --no-masked-softmax-fusion \
           --tensor-model-parallel-size 2 \
           --pipeline-model-parallel-size 1 
"

python -m torch.distributed.launch --nproc_per_node=8 --master_port 9997 train_megatron_tp.py $MEGATRON_ARGS \
--train_batch_size 1 \
--vocab_size 30522 \
--hidden_size 1024 \
--num_hidden_layers 24 \
--num_attention_heads 16 \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 1 \
--profile 1 \
--tensor-model-parallel-size 8
