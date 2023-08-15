PYTHONPATH=/home/pkuhetu/envs/miniconda3/envs/torch201/bin/python
$PYTHONPATH -m torch.distributed.launch --nproc_per_node=8 --master_port 9997 backward_correctness_test.py \
--train_batch_size 8 \
--vocab_size 30522 \
--hidden_size 2 \
--num_hidden_layers 24 \
--num_attention_heads 1 \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--dropout_prob 0 \
--profile 0 \
--no-gradient-accumulation-fusion \
--no-async-tensor-model-parallel-allreduce \
