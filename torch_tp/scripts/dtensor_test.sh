PYTHONPATH=python
$PYTHONPATH -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 dtensor_test.py
