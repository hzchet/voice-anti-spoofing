NAME?=spoof
GPUS?=1
DATA_DIR?=/hdd/aidar/as_data_dir
NOTEBOOKS?=/hdd/aidar/notebooks/anti-spoofing
SAVE_DIR?=/hdd/aidar/as_save_dir
USER?=$(shell whoami)
UID?=$(shell id -u)
GID?=$(shell id -g)

.PHONY: build run

build:
	docker build \
	--build-arg UID=$(UID) \
	--build-arg GID=$(GID) \
	--build-arg UNAME=$(USER) \
	-t $(NAME) .

run:
	docker run --rm -it --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=$(GPUS) \
	--ipc=host \
	--net=host \
	-v $(PWD):/workspace \
	-v $(DATA_DIR):/workspace/data \
	-v $(NOTEBOOKS):/workspace/notebooks \
	-v $(SAVE_DIR):/workspace/saved \
	--name=$(NAME) \
	$(NAME) \
	bash
