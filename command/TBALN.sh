# # MSCOCO14 ResNet101 448 bs 126
# torchrun --nnodes=1 --nproc_per_node=6 --rdzv_id=100 --rdzv_backend=c10d \
# /root/nas-private/TBALN/main.py -a 'TBALN-R101-448' \
# --dataset_dir '/root/nas-private/Dataset' \
# --backbone resnet101 --dataname coco14 --batch-size 126 --print-freq 400 \
# --output './output/ResNet_448_MSCOCO14/' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
# --gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
# --epochs 80 --lr 1e-4 --optim AdamW --pretrained \
# --num_class 80 --img_size 448 --weight-decay 1e-2 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --early-stop --amp \
# --ema-decay 0.9998 \
# --gpus 0,1,2,3,4,5

# MSCOCO14 ResNet101 448 bs 8
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
