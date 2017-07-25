## Sketch Simplification using CycleGAN with Synthetic Dataset

This repo is built during [MLJejuCamp2017](https://github.com/MLJejuCamp2017).

## Usage
1. Training
    ```shell
    cd $(PROJECT_ROOT)/srcs
    python main.py --model_type our_cycle_gan
    ```

2. Testing
    ```shell
    cd $(PROJECT_ROOT)/srcs
    python main.py --mode test --saved_model_path checkpoints/model_path/model.ckpt-7000 \
        --test_size 512
    ```
