# MSCOCO14 ResNet101 448 bs 64
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
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v3 \
--early-stop --amp \
--seed 1015 \
--ema-decay 0.9998 \
--gpus 0,1,2,3,4,5

# # MSCOCO14 ResNet101 448 bs 64
# torchrun --nnodes=1 --nproc_per_node=6 --rdzv_id=100 --rdzv_backend=c10d \
# /root/nas-private/Q2L/main.py -a 'Q2L-R101-576' \
# --dataset_dir '/root/nas-private/Dataset' \
# --backbone resnet101 --dataname coco14 --batch-size 126 --print-freq 400 \
# --output './output/ResNet_576_MSCOCO14/' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
# --gamma_pos 0 --gamma_neg 2 --dtgfl --loss_clip 0 \
# --epochs 80 --lr 1e-4 --optim AdamW --pretrained \
# --num_class 80 --img_size 448 --weight-decay 1e-2 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --hidden_dim 2048 --dim_feedforward 8192 \
# --enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v3 \
# --early-stop --amp \
# --ema-decay 0.9998 \
# --gpus 0,1,2,3,4,5

# # VOC2007 ResNet101 448 bs 66 
# torchrun --nnodes=1 --nproc_per_node=6 --rdzv_id=100 --rdzv_backend=c10d \
# /root/nas-private/Q2L/main.py -a 'Q2L-R101-448' \
# --dataset_dir '/root/nas-private/Dataset' \
# --backbone resnet101 --dataname voc2007 --batch-size 66 --print-freq 40 \
# --output './output/ResNet_448_voc2007/' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
# --gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
# --epochs 80 --lr 0.0001 --optim AdamW --pretrained \
# --num_class 20 --img_size 448 --weight-decay 0.01 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --hidden_dim 2048 --dim_feedforward 8192 \
# --resume '/root/nas-private/Q2L/command/output/ResNet_576_MSCOCO14/coco14_resnet101_bs180_e1-d2_asl-0-4-00_lr000015_lrp01_wd0015_AdamW_crop_amp_/model_best.pth.tar' \
# --resume_omit  query_embed_1.weight query_embed_2.weight query_embed_3.weight fc.W fc.b \
# --enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v3 \
# --early-stop --amp \
# --ema-decay 0.9997 \
# --out_aps \
# --gpus 0,1,2,3,4,5

# # VOC2007 ResNet101 448 bs 66 
# torchrun --nnodes=1 --nproc_per_node=6 --rdzv_id=100 --rdzv_backend=c10d \
# /root/nas-private/Q2L/main.py -a 'Q2L-R101-448' \
# --dataset_dir '/root/nas-private/Dataset' \
# --backbone resnet101 --dataname voc2007 --batch-size 18 --print-freq 40 \
# --output './output/ResNet_448_voc2007/' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
# --gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
# --epochs 80 --lr 0.0001 --optim Adam_twd --pretrained \
# --num_class 20 --img_size 448 --weight-decay 0.01 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --hidden_dim 2048 --dim_feedforward 8192 \
# --enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v3 \
# --early-stop --amp \
# --ema-decay 0.9997 \
# --out_aps \
# --seed 1015 \
# --gpus 0,1,2,3,4,5

# # VOC2007 ResNet101 448 bs 66 
# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
# /root/nas-private/Q2L/main.py -a 'Q2L-R101-448' \
# --dataset_dir '/root/nas-private/Dataset' \
# --backbone resnet101 --dataname voc2007 --batch-size 16 --print-freq 40 \
# --output './output/ResNet_448_voc2007/' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
# --gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
# --epochs 80 --lr 5e-5 --optim AdamW --pretrained \
# --num_class 20 --img_size 448 --weight-decay 5e-3 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --hidden_dim 2048 --dim_feedforward 8192 \
# --resume '/root/nas-private/Q2L/command/output/ResNet_448_MSCOCO14/85.14 2 3 coco14_resnet101_bs126_e1-d2_asl-0-4-00_lr00001_lrp01_wd001_AdamW_crop_amp_/model_best.pth.tar' \
# --resume_omit  query_embed_1.weight query_embed_2.weight query_embed_3.weight fc.W fc.b \
# --enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v3 \
# --early-stop --amp \
# --ema-decay 0.9997 \
# --out_aps \
# --seed 1025 \
# --gpus 0,1

# # VOC2007 ResNet101 448 bs 66 
# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
# /root/nas-private/Q2L/main.py -a 'Q2L-R101-448' \
# --dataset_dir '/root/nas-private/Dataset' \
# --backbone resnet101 --dataname voc2012 --batch-size 16 --print-freq 40 \
# --output './output/ResNet_448_voc2012/' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
# --gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
# --epochs 80 --lr 5e-5 --optim AdamW --pretrained \
# --num_class 20 --img_size 448 --weight-decay 5e-3 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --hidden_dim 2048 --dim_feedforward 8192 \
# --resume '/root/nas-private/Q2L/command/output/ResNet_448_MSCOCO14/85.14 2 3 coco14_resnet101_bs126_e1-d2_asl-0-4-00_lr00001_lrp01_wd001_AdamW_crop_amp_/model_best.pth.tar' \
# --resume_omit  query_embed_1.weight query_embed_2.weight query_embed_3.weight fc.W fc.b \
# --enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v3 \
# --early-stop --amp \
# --ema-decay 0.9997 \
# --out_aps \
# --seed 3407 \
# --gpus 0,1