python -m torch.distributed.launch --nproc_per_node=8 --master_port 9997 --use_env train_torch_tp.py
