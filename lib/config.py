import argparse
from pathlib import Path
from ast import arg

def parser_args():
    available_models = ['TBALN-R101-448', 'TBALN-R101-576']
    parser = argparse.ArgumentParser(description='Query2Label Training')
    parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14', 'voc2007', 'voc2012', 'nus_wide', 'vg500', 'vg1000'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/media/data2/MLICdataset/')
    parser.add_argument('--img_size', default=384, type=int,
                        help='size of input images')

    parser.add_argument('--output', metavar='DIR', default='/root/nas-private/TBALN/output',
                        help='path to output folder')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--frozen_backbone', action='store_true', default=False,
                        help='apply frozen backbone in train')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='TBALN-R101-448',
                        choices=available_models,
                        help='model architecture: ' +' | '.join(available_models) +
                            ' (default: TBALN-R101-448)')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')    
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')    


    parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lrp', '--learning-rate-backbone', default=0.1, type=float,
                        metavar='LR', help='initial learning rate of backbone', dest='lrp')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--crop', action='store_true', default=False,
                        help='apply multi scale crop')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')
    parser.add_argument('--remove_norm', action='store_true', default=False,
                        help='remove normalize')
    # parser.add_argument('--mix_up', action='store_true', default=False,
    #                     help='mix up for images')


    # * Transformer
    # You can modify the annotated sections in the TBALN.py file.
    # parser.add_argument('--enc_layers', default=1, type=int, 
    #                     help="Number of encoding layers in the transformer")
    # parser.add_argument('--dec_layers', default=2, type=int,
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--dim_feedforward', default=1024, type=int,
    #                     help="Intermediate size of the feedforward layers in the transformer blocks")
    # parser.add_argument('--hidden_dim', default=1024, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    # parser.add_argument('--nheads', default=4, type=int,
    #                     help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true',default= False)
    # parser.add_argument('--position_embedding', default='v3', type=str, choices=('sine', 'v2', 'v3'),
    #                     help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * raining
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')

    # display
    parser.add_argument('--out_aps', action='store_true', default=False,
                        help='display the out of aps')

    # GPU
    parser.add_argument('--gpus', default='0', type=str,
                        help='select GPUS (default: 0)')

    args = parser.parse_args()

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)


    output = str(args.dataname)+'_'+str(args.backbone)+'_'
    output += 'bs{}_'.format(args.batch_size)
    output += 'asl-{}-{}-{}_'.format(str(int(args.gamma_pos)), str(int(args.gamma_neg)), str(args.loss_clip)).replace('.', '')
    if args.orid_norm:
        output += 'norm_'
    output += 'lr{}_lrp{}_wd{}_{}_'.format(args.lr, args.lrp, args.weight_decay, args.optim).replace('.', '')

    if args.cutout:
        output += 'crop_'
    if args.crop:
        output += 'cut-{}-{}_{}_'.format(args.n_holes, args.length, args.img_size)
    if args.amp:
        output += 'amp_'
    # output_mc6/nuswide_mlter-tresl-e1-d2_slasl2-0-1-0-norm_lr1e-4_1e-4_Adam_twd_1e-2_lclr_cut-1-224_448_amp_
    import os
    args.output = os.path.join(args.output, output)
    # print(args.output)

    return args