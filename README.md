# Checkpointless training on Amazon SageMaker HyperPod

Checkpointless training on Amazon SageMaker HyperPod eliminates disruptive checkpoint-restart cycles, maintaining forward training momentum despite failures, reducing recovery time from hours to minutes.

## Key Features

- **In-Process Recovery**: Recover from node failures in minutes without losing training progress by using redundant model copies stored in GPU memory
- **Fast Initialization**: Accelerate training restarts by bypassing expensive communication (NCCL/Gloo) setup processes
- **Smart Data Caching**: Pre-load and cache training data batches to eliminate delays when resuming training after failures
- **Built-in Redundancy**: Leverage distributed optimizer instances for checkpointless recovery
- **NeMo Integration**: Works seamlessly with PyTorch Lightning and NVIDIA NeMo toolkit for large language model training

## Getting Started Examples

| Model | Method | Size | Nodes | Instance | Accelerator | Recipe | Script |
|-------|--------|------|-------|----------|-------------|--------|--------|
| GPT OSS | Full finetune example | 120b | 16 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/fine-tuning/gpt_oss/checkpointless_gpt_oss_120b_full_fine_tuning.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/gpt_oss/run_checkpointless_gpt_oss_120b_full_fine_tuning.sh) |
| GPT OSS | LoRA-example | 120b | 2 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/fine-tuning/gpt_oss/checkpointless_gpt_oss_120b_lora.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/gpt_oss/run_checkpointless_gpt_oss_120b_lora.sh) |
| Llama3 | Pretrain example | 70b | 16 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/training/llama/checkpointless_llama3_70b_pretrain.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/llama/run_checkpointless_llama3_70b_pretrain.sh) |
| Llama3 | LoRA-example | 70b | 2 | p5.48xlarge | GPU H100 | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/recipes_collection/recipes/fine-tuning/llama/checkpointless_llama3_70b_lora.yaml) | [link](https://github.com/aws/sagemaker-hyperpod-recipes/tree/main/launcher_scripts/llama/run_checkpointless_llama3_70b_lora.sh) |

## User Guide

For comprehensive documentation including installation steps, environment setup, configuration options, and detailed usage examples, see the tutorials at [Amazon SageMaker HyperPod Checkpointless training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-checkpointless.html)..

## Quick Start Guide

### Launch Training

#### Hyperpod Recipe Launcher

You can use the SageMaker HyperPod recipes to submit your training job. Using the recipes involves updating k8s.yaml, config.yaml and running the launch script.

```bash
bash launcher_scripts/gpt_oss/run_checkpointless_nemo_gpt_oss_120b_fine_tuning.sh
```

#### Launch Using kubectl

Alternatively, you can deploy the training job directly using kubectl:

```bash
kubectl apply -f <path_to_config>.yaml
```

#### Monitor Job Status

```bash
kubectl get pods
kubectl logs <pod-name>
```

For detailed installation steps, environment setup, and configuration options, see the tutorials at [Amazon SageMaker HyperPod Checkpointless training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-checkpointless.html).

## Recommended Requirements

| Component | Version |
|-----------|---------|
| Python | >=3.12 |
| PyTorch | >=2.6.0 |
| NeMo Toolkit | 2.6.0rc0 |
| CUDA | 12.5+ |
| Infrastructure | AWS HyperPod Kubernetes cluster |
| Storage | Shared storage (FSx/NFS) |

---

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
**Note**: This repository is temporarily not accepting pull requests.

## License

This project is licensed under the Apache-2.0 License.