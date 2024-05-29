import torch
import time
import os
import numpy as np
import _init_paths
import torch.distributed as dist
import torch.nn as nn
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import voc_mAP

@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
        # for i, (images, target, iamge_name) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                with torch.no_grad():
                    # output, _, _ = model(images)
                    output = model(images)
                    loss = criterion(output, target)
                    if args.loss_dev > 0:
                        loss *= args.loss_dev
                    output_sm = torch.sigmoid(output)
                    if torch.isnan(loss):
                        saveflag = True

            # test_for_voc2012(output, iamge_name, args)        
                    
            # record loss
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data
            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            # del output_sm
            # del target
            saved_data.append(_item)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )
        
        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating mAP:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP                
            mAP, aps ,OP,OR,OF1,CP,CR,CF1 = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
          
            logger.info("  mAP: {}".format(mAP))
            logger.info("  OP: {}".format(OP))
            logger.info("  OR: {}".format(OR))
            logger.info("  OF1: {}".format(OF1))
            logger.info("  CP: {}".format(CP))
            logger.info("  CR: {}".format(CR))
            logger.info("  CF1: {}".format(CF1))
            if args.out_aps:
                logger.info("  aps: {}".format(np.array2string(aps, precision=5)))
        else:
            mAP = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, mAP

def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

def test_for_voc2012(output, iamge_name, args):
    if not os.path.exists(os.path.join(args.output, 'results', 'VOC2012', 'Main')):
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Main'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Layout'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Action'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'results', 'VOC2012', 'Segmentation'), exist_ok=True)

    temp = output.cpu().data.numpy()
    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

    # temp = output_sm.cpu().data.numpy()
    temp = output.cpu().data.numpy()
    for row in range(temp.shape[0]):
        for col in range(temp.shape[1]):
            name = iamge_name[row]
            val = temp[row, col]
            temp_categories_file = os.path.join(args.output, 'results', 'VOC2012', 'Main', 'comp1_cls_test_' + object_categories[col] + '.txt')
            with open(temp_categories_file, 'a') as file:
                file.write(name + ' ' + str(val) + '\n')