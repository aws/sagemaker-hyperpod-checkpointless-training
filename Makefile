.PHONY: all run local build test install

script = ${PWD}/examples/inprocess/sagemaker/hp_basic_example.py
pp_script = ${PWD}/examples/inprocess/sagemaker/hp_pp_example.py
sqsh ?= ${PWD}/hyperpod-checkpointless-training+latest.sqsh

all: run

run:
	${PWD}/examples/inprocess/sagemaker/hprun.sh \
		-i $(sqsh) \
		$(script) \
		--device cuda \
		--log-interval 5 \
		--path ${PWD} \
		--fault-prob 0.001 \
		--size 256 \
		--layers 16 \
		--total-iterations 10000

local:
	hprun --nproc_per_node=8 $(script) --device cuda --log-interval 5 --total-iterations 1000

build:
	${PWD}/examples/inprocess/sagemaker/build.sh -n hyperpod-checkpointless-training -f ${PWD}/examples/inprocess/sagemaker/Dockerfile

test:
	HPWRAPPER_LOG_LEVEL=debug pytest tests/inprocess/sagemaker/ -s

install:
	pip install --force-reinstall --no-deps .
