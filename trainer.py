import os
import time
import shutil
import numpy as np
import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
from utils import distributed_all_gather, AverageMeter
from tempWatch import Alarm
from time import sleep
import torch.utils.data.distributed
from monai.data import decollate_batch
import pdb


def train_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                args,
                logger):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    if args.modality == 'multi':
        for idx, batch_data in enumerate(loader):
            data, target,digits,name = batch_data
            digits = digits.squeeze(dim=1)
            data, target,digits = data.cuda(), target.cuda(),digits.cuda()
            for param in model.parameters(): param.grad = None
            with autocast(enabled=args.amp):
                logits = model([data, digits])
                loss = loss_func(logits, target) #,model)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            run_loss.update(loss.item(), n=args.batch_size)
            # logger.info(f'Epoch {epoch}/{args.max_epochs} {idx}/{len(loader)} loss:{run_loss.avg} time {time.time() - start_time}s')
            logger.info('Epoch {}/{} {}/{} loss:{:.4f} time {:.2f}s'.format(epoch, args.max_epochs, idx, len(loader),
                                                                            run_loss.avg, time.time() - start_time))
            alarm, des = Alarm()
            if alarm == 1:
                logger.info(f'Current temprature is out of control due to {des}')
                while alarm:
                    sleep(args.sleep)
                    alarm, _ = Alarm()
    
    return run_loss.avg


def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              y_trans=None,
              y_pred_trans=None,
              logger=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    metric_values=[]
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32).cuda()
        y = torch.tensor([], dtype=torch.long).cuda()
        if args.modality=='multi':
            for idx, batch_data in enumerate(loader):
                data, target,digits,name = batch_data
                digits=digits.squeeze(dim=1)
                data, target,digits = data.cuda(), target.cuda(),digits.cuda()
                y_pred = torch.cat([y_pred, model(data,digits)], dim=0)
                y = torch.cat([y, target], dim=0)        
        y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        acc_func(y_pred_act, y_onehot)
        result = acc_func.aggregate()
        acc_func.reset()
        del y_pred_act, y_onehot
        metric_values.append(result)
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        logger.info(f'Val {epoch}/{args.max_epochs} '
                    f', accuracy: {acc_metric}'
                    f', time {time.time() - start_time}s')
    return acc_metric

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None,
                    logger=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename = os.path.join(args.result_root, filename)
    torch.save(save_dict, filename)
    logger.info(f'Saving checkpoint {filename}')

def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 y_trans=None,
                 y_pred_trans=None,
                 semantic_classes=None,
                 logger=None):
    writer = None
    if args.result_root is not None:
        writer = SummaryWriter(log_dir=args.result_root)
    logger.info(f'Writing Tensorboard logs to {args.result_root}')
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.
    logger.info('Begin Training' + '-' * 70)
    for epoch in range(start_epoch, args.max_epochs):
        logger.info(f'{time.ctime()}, Epoch:, {epoch}')
        epoch_time = time.time()
        if writer is not None:
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        try:
            train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 args=args,
                                 logger=logger)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

        logger.info(f'Final training  {epoch}/{args.max_epochs - 1}, loss: {train_loss},time {time.time() - epoch_time}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False

        if (epoch+1) % args.save_every == 0 and args.result_root is not None and args.save_checkpoint:
            save_checkpoint(model,
                        epoch,
                        args,
                        best_acc=0,
                        filename=fr'model_{epoch}.pt',logger=logger)


        if (epoch+1) % args.val_every == 0:
            # if args.distributed:
            #     torch.distributed.barrier()
            epoch_time = time.time()

            val_acc = val_epoch(model,
                                val_loader,
                                epoch=epoch,
                                acc_func=acc_func,
                                args=args,
                                y_trans=y_trans,
                                y_pred_trans=y_pred_trans,
                                logger=logger
                                )
            # revised by debbie
            if writer is not None:
                writer.add_scalar("val_accuracy", val_acc, epoch )
            if val_acc > val_acc_max:
                    logger.info('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_acc))
                    val_acc_max = val_acc
                    b_new_best = True
                    if args.result_root is not None:
                        save_checkpoint(model, epoch, args,
                                        best_acc=val_acc_max,
                                        filename='model_best.pt',
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        logger=logger)

            if args.result_root is not None:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_final.pt',
                                logger=logger)
                if b_new_best:
                        logger.info('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(args.result_root, 'model_final.pt'), os.path.join(args.result_root, 'model_best.pt'))
                        b_new_best = False

        if scheduler is not None:
            scheduler.step()

    logger.info('Training Finished !, Best Accuracy: {:.4f}'.format(val_acc_max))

    return val_acc_max
