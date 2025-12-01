# OSS llama3 model in slurm with one node

### Setup

Step1: Pull from NGC container
```
docker pull nvcr.io/nvidia/nemo:25.07

```

Step2: Launch it using
```
docker run   --gpus all -v /fsx/:/fsx  -it   --rm   --shm-size=16g   --ulimit memlock=-1   --ulimit stack=67108864 nvcr.io/nvidia/nemo:25.07
```

Step3: Get into the docker and run
```
torchrun --nproc_per_node 8 deepseek_v3_pretrain.py
```

