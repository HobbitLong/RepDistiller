# RepDistiller

This repo:

(1) covers the implementation of the following paper:

"Contrastive Representation Distillation" (CRD)

(2) benchmarks the state-of-the-art knowledge distillation methods in Pytorch


## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
    Therefore, the command for running CRD is something like:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1
    ```
    
    Combining a distillation objective with KD is simply done by setting `-a` as a non-zero value, which results in the following example (combining CRD with KD)
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1     
    ```

3. (optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`

