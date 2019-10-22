# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# kd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
# FitNet
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8x4 -a 0 -b 100 --trial 1
# AT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8x4 -a 0 -b 1000 --trial 1
# SP
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8x4 -a 0 -b 3000 --trial 1
# CC
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8x4 -a 0 -b 0.02 --trial 1
# VID
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8x4 -a 0 -b 1 --trial 1
# RKD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -a 0 -b 1 --trial 1
# PKT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4 -a 0 -b 30000 --trial 1
# AB
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8x4 -a 0 -b 1 --trial 1
# FT
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8x4 -a 0 -b 200 --trial 1
# FSP
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8x4 -a 0 -b 50 --trial 1
# NST
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8x4 -a 0 -b 50 --trial 1
# CRD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1

# CRD+KD
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1