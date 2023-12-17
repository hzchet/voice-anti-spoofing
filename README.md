# Voice Anti-spoofing

This repository contains implementation of the [LightCNN (LCNN)](https://arxiv.org/abs/1511.02683) and [RawNet2](https://arxiv.org/abs/2011.01108) countermeasure systems.

# Reproducing results
To reproduce trainig of the final model, follow the steps:

1. Specify the `GPUS` (gpu indices that will be used during training), `SAVE_DIR` (directory where all the logs & checkpoints will be stored), `DATA_DIR` (directory that will store the training data), `NOTEBOOK_DIR` (directory that contains your notebooks, for debugging purposes) in the `Makefile`. Set up `WANDB_API_KEY` variable in the `Dockerfile` to log the training process.

2. Build and run the `Docker` container
```bash
make build && make run
```

3. Run the pipeline described in the `configs/rawnet_config.yaml` configuration file.
```bash
python3 train.py --config-name rawnet_config.yaml
```

# Running tests
1. In order to run an inference on pre-trained model, you should first download its weights by running
```bash
python3 install_weights.py
```

2. Run
```bash
python3 test.py --config-name rawnet_config.yaml +resume="saved/models/final/weight.pth"
```
