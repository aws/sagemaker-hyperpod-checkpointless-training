#!/bin/bash

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
IMAGE=""
PROGRAM="$0"
GPUS=8
NSYS_PATH=""

err() {
  echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2
}

usage() {
  set +x
  cat <<EOF
usage:  $PROGRAM [OPTIONS] params

options:

  -h,--help             show this help
  -i,--image [image]    docker image for the experiment
  -g,--gpus [gpus]      number of GPUs to be used
  -n,--nsys [path]      NSight output path

EOF
}

cleanup() {
  srun bash -c 'docker ps -q -a | xargs -I{} docker rm -f {}' || true
  srun bash -c "enroot list | xargs -I{} enroot remove --force {}" || true
}

run() {
  set -exo pipefail

  local root_dir="${DIR}/../../../"
  local image="${1}"
  local gpus="${2}"
  local nsys_path="${3}"
  local training_args=("${@:4}")
  local args=()
  local hprun_cmd="hprun"
  local mountpoints=()
  local container_mounts
  local nsys_cmd=()
  local cmd

  if ! [ -f "${image}" ]; then
    err "enroot image (.sqsh) ${image} not found"
    return 1
  fi

  if [ -z "$SLURM_NNODES" ]; then
    err "please launch your job via Slurm"
    return 1
  fi

  if [ -n "${nsys_path}" ]; then
    nsys_cmd+=(nsys profile -w true)
    nsys_cmd+=(-t "cuda,nvtx,osrt,cudnn,cublas")
    nsys_cmd+=(--gpu-metrics-device=all)
    nsys_cmd+=(--nic-metrics=true)
    nsys_cmd+=(--capture-range=cudaProfilerApi)
    nsys_cmd+=(--capture-range-end=stop)
    nsys_cmd+=(--cuda-memory-usage=true)
    nsys_cmd+=(--cudabacktrace=true)
    nsys_cmd+=(-x true)
    nsys_cmd+=(-o "$nsys_path")
    nsys_cmd+=(--force-overwrite=true)
  fi

  args+=(--nproc_per_node="${gpus}")
  args+=(--rdzv_id=100)
  args+=(--rdzv_backend=c10d)
  args+=(--nnodes="$SLURM_NNODES")
  args+=(--rdzv_endpoint="$(srun hostname | uniq | sort | head -n 1):29400")

  mountpoints+=("/fsx:/fsx")
  mountpoints+=("/var/log/aws/clusters:/var/log/aws/clusters")
  container_mounts="$(IFS=','; echo "${mountpoints[*]}")"
  root_dir="$(realpath "${root_dir}")"

  cmd="$(cat <<EOF
export NCCL_SOCKET_IFNAME="^lo,docker,veth_def_agent"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG="INFO"
export GPU_NUM_DEVICES=${GPUS}
export PYTHONPATH="${root_dir}/src:\$PYTHONPATH"
${nsys_cmd[@]} ${hprun_cmd} ${args[@]} ${training_args[@]}
EOF
)"

  srun --container-image "${image}"  \
    --container-mounts "${container_mounts}" \
    bash -c "${cmd}"
}

while (( "$#" )); do
  case "$1" in
    -h|-\?|--help) usage; exit 0 ;;
    -i|--image) IMAGE="$2"; shift 2 ;;
    -g|--gpus) GPUS="${2}"; shift 2 ;;
    -n|--nsys) NSYS_PATH="${2}"; shift 2 ;;
    *) break
  esac
done

cleanup

if ! run "${IMAGE}" "${GPUS}" "${NSYS_PATH}" "$@"; then
  err "run enroot hprun fail"
  exit 1
fi
