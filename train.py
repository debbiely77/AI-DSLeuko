import os
import PIL
import argparse
import logging
import sys
from dataset import MedNISTDataset
from datetime import datetime
import numpy as np
from trainer import run_training
from parser import Parser
from getLogger import get_logger
from model_up import *
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)

parser = argparse.ArgumentParser(description='Classification')
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--cfg', default='config', type=str)
args = parser.parse_args()
args = Parser(args.cfg).add_args(args)
SAVE_PATH = os.path.join(os.getcwd(), 'runs', args.model)

def L2Loss(models, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in models.named_parameters():
        if 'bias' not in name and parma.requires_grad==True:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

def L1Loss(models, beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in models.named_parameters():
        if 'bias' not in name and parma.requires_grad==True:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma))
    return l1_loss

def focal_loss_with_regularization(y_pred, y_true,models):
    focal = FocalLoss()(y_pred, y_true)
    l2_loss = L2Loss(models, 0.001)
    l1_loss = L1Loss(models, 0.001)
    total_loss = focal + l2_loss + l1_loss
    return total_loss

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


data_dir=args.data_dir
class_names=['non-APL AML','APL','ALL','CP-CML','BCRABL1-MPN','CLL','BacterialInfection','ViralInfection']
#
num_class  = len(class_names)
args.num_classes=num_class
# 
image_files = [
    [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))
    if x.endswith(".png")]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []#
image_class = []#
for i in range(num_class): #
    image_files_list.extend(image_files[i])#
    image_class.extend([i] * num_each[i])#
num_total = len(image_class) #
image_width, image_height = PIL.Image.open(image_files_list[0]).size #

print(f"Total image count: {num_total}")#
print(f"Image dimensions: {image_width} x {image_height}")#
print(f"Label names: {class_names}")#
print(f"Label counts: {num_each}") #

val_frac = 0.1
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

val_split = int(val_frac * length)
val_indices = indices[:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]


print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}")

train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.0, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.0),
        RandFlip(spatial_axis=1, prob=0.0),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ]
)
val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])

if args.modality=='multi':
    train_ds = MedNISTDataset(train_x, train_y, train_transforms,digit_root=args.digit_root)
    val_ds = MedNISTDataset(val_x, val_y, val_transforms,digit_root=args.digit_root)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,collate_fn=pad_list_data_collate)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2,collate_fn=pad_list_data_collate)

# val_interval = 10
if args.modality=='multi':
    if args.model == 'Resnet_FM':
        model = Resnet_FM(args)
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.cuda()

# create a temporary directory and 40 random image, mask pairs
dt = datetime.now()
temp_root = 'fold_{}_'.format(args.fold)+'batchsize_{}_'.format(args.batch_size) + 'time_' + dt.strftime('%y-%m-%d-%H:%M:%S')
result_root = os.path.join(SAVE_PATH,
                           temp_root.replace(':', '-').replace(' ', '').replace('\'', '').replace('[', '').replace(
                               ']', ''))
args.result_root = result_root
if not os.path.exists(result_root):
    os.makedirs(result_root)

# Create main logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = get_logger(os.path.join(result_root, 'training-setting.log'), logging.DEBUG)
# logger.info(f'Training of {args.net}：')
logger.info(args)
logger.info(class_names)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Total parameters count: {pytorch_total_params}')

best_acc = 0
start_epoch = 0

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace('backbone.', '')] = v
    model.load_state_dict(new_state_dict, strict=False)
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    if 'best_acc' in checkpoint:
        best_acc = checkpoint['best_acc']
    logger.info("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

if args.loss=='CrossEntropy':
    loss_function = torch.nn.CrossEntropyLoss()
elif args.loss=='Focal':
    # loss_function =FocalLoss()
    loss_function=focal_loss_with_regularization

if args.optim_name == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.optim_lr,
                                 weight_decay=args.reg_weight)
elif args.optim_name == 'adamw':
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.optim_lr,
                                  weight_decay=args.reg_weight)
elif args.optim_name == 'sgd':
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        # model.parameters(),
                                lr=args.optim_lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.reg_weight)
else:
    raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

if args.lrschedule == 'cosine_anneal':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_epochs)
    if args.checkpoint is not None:
        scheduler.step(epoch=start_epoch)
else:
    scheduler = None
auc_metric = ROCAUCMetric()
accuracy = run_training(model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        loss_func=loss_function,
                        acc_func=auc_metric,
                        args=args,
                        scheduler=scheduler,
                        logger=logger,
                        y_trans=y_trans,
                        y_pred_trans=y_pred_trans,
                        start_epoch=start_epoch
)

