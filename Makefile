## training mode (newly)
train:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python wgan.py train --name $(shell date "+%Y-%m-%d-%s")

.DEFAULT_GOAL := help

## shows this
help:
	@grep -A1 '^## ' ${MAKEFILE_LIST} | grep -v '^--' |\
		sed 's/^## *//g; s/:$$//g' |\
		awk 'NR % 2 == 1 { PREV=$$0 } NR % 2 == 0 { printf "\033[32m%-18s\033[0m %s\n", $$0, PREV }'
