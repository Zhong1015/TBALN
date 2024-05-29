# TBALN
论文"基于多尺度特征融合和Transformer的多标签图像分类"的官方PyTorch实现

## 实验结果
![coco](./COCO.png)

## 实验环境
Python = 3.8
Pytorch = 1.10.1
Numpy = 1.23.1 
tqdm = 4.63.0
yaml = 0.2.5
tkinter = 8.6.11

## 模型权重下载
我们提供了COCO2014数据集下的模型权重文件，以及相关的日志文件：https://drive.google.com/drive/folders/15M2KiguWuvmptFVEaAa7HVgL1_h4CYQP?usp=drive_link

## 训练
```sh
# 单卡
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d \
/root/nas-private/TBALN/main.py -a 'TBALN-R101-448' \
--dataset_dir '/root/nas-private/Dataset' \
--backbone resnet101 --dataname coco14 --batch-size 8 --print-freq 400 \
--output './output/ResNet_448_MSCOCO14/' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--early-stop --amp \
--ema-decay 0.9998 \
--gpus 0


# 多卡
torchrun --nnodes=1 --nproc_per_node=6 --rdzv_id=100 --rdzv_backend=c10d \
/root/nas-private/Q2L/main.py -a 'Q2L-R101-448' \
--dataset_dir '/root/nas-private/Dataset' \
--backbone resnet101 --dataname coco14 --batch-size 126 --print-freq 400 \
--output './output/ResNet_448_MSCOCO14/' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--early-stop --amp \
--ema-decay 0.9998 \
--gpus 0,1,2,3,4,5
'''
## 致谢
我们感谢[Q2L]([https://github.com/Alibaba-MIIL/ASL](https://github.com/SlongLiu/query2labels)), [SADCL](https://github.com/yu-gi-oh-leilei/SADCL), [detr](https://github.com/facebookresearch/detr)的杰出工作和代码.
