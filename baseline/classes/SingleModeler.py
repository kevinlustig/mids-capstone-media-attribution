import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from classes.CombinedModel import CombinedModel
from classes.ImageFolderWithData import ImageFolderWithData
from classes.utils import AverageMeter
from classes.utils import ProgressMeter
from classes.utils import ModelOptions

cudnn.deterministic = True

opts = ModelOptions(**{
    'start_epoch':0,
    'num_classes':3592,
    'epochs':8,
    'lr':.05,
    'momentum':0.9,
    'weight_decay':5e-4,
    'print_freq':50,
    'batch_size':8,#10,
    'workers':0,
    'traindir':"train",
    'valdir':"val",
    'image_size':224,
    ##Derived from processing in dataset.ipynb
    'rgb_mean':[0.3869, 0.4162, 0.4315],
    'rgb_std':[0.3015, 0.2849, 0.2924],
    'checkpoint_path': "./checkpoint.pth.tar",
    #'arch': "efficientnet_v2_s"
    #'arch': "resnext101_64x4d"
    'arch': "densenet201"
    #'arch': 'efficientnet'
})

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SingleModeler():
  def __init__(self):
    self.writer = SummaryWriter('logs/' + opts.arch + '/' + datetime.now().strftime("%s"))

  def init(self):
    print("Initializing single-device training")

    gpu = 0

    self.setup_dataset()

    #self.model = CombinedModel(gpu,opts.arch,self.train_dataset.geo_classes,self.train_dataset.date_classes,opts.num_classes).models['image']
    #self.model = CombinedModel(gpu,opts.arch,self.train_dataset.geo_classes,self.train_dataset.date_classes,opts.num_classes)
    self.model = CombinedModel(gpu,opts.arch,opts.num_classes)

    torch.cuda.set_device(gpu)
    self.model.cuda(gpu)

    self.criterion = nn.CrossEntropyLoss().cuda(gpu)
    self.scaler = GradScaler()
    self.optimizer = optim.SGD(
        self.model.parameters(),
        lr=opts.lr,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )
    
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=len(self.train_loader),eta_min=.00001)
    
    self.run(gpu)

  def setup_dataset(self):
    #Training
    self.train_dataset = ImageFolderWithData(opts.traindir, data_file='./photo_metadata.csv', transform=transforms.Compose([
      transforms.RandomResizedCrop(opts.image_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(opts.rgb_mean, opts.rgb_std)
    ]))

    self.train_loader = torch.utils.data.DataLoader(
      dataset=self.train_dataset,
      batch_size=opts.batch_size,
      shuffle=True,            
      num_workers=opts.workers,
      pin_memory=True
    )

    #Validation
    self.val_dataset = ImageFolderWithData(opts.valdir, data_file='./photo_metadata.csv', transform=transforms.Compose([
        transforms.Resize(opts.image_size),
        transforms.CenterCrop(opts.image_size),
        transforms.ToTensor(),
        transforms.Normalize(opts.rgb_mean, opts.rgb_std)
    ]))
    self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=opts.batch_size, shuffle=True) 

  def run(self,gpu):
    print("Beginning training loops")
    start = datetime.now()
    self.global_step = 0
    for epoch in range(opts.start_epoch, opts.epochs):
        self.train(gpu,epoch)
        self.validate(gpu,epoch)

        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer_dict' : self.optimizer.state_dict(),
        }, self.model, filename=opts.checkpoint_path, gpu=gpu)

        print('Learning rate: ' + str(self.scheduler.get_last_lr()))
        print("Time elapsed: " + str(datetime.now() - start))

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

  def train(self,gpu,epoch):
    print("Training")
    self.create_meters(self.train_loader,f"Epoch: [{epoch}]")

    self.model.train()

    # for i, (images, target, geo, date) in enumerate(self.train_loader):
    for i, (images, target) in enumerate(self.train_loader):
      start = time.time() 

      #if i == 0:
      #  print(date)

      images = images.cuda(non_blocking=True)
      # geo = geo.cuda(non_blocking=True)
      # date = date.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      self.optimizer.zero_grad()
      with autocast():
        #output = self.model(images,geo,date)
        #output = self.model(images,date)
        output = self.model(images)
        loss = self.criterion(output, target)

      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()

      self.scheduler.step(self.global_step)
      self.global_step += 1

      self.update_meters(start, output, target, loss, images)
      
      if gpu == 0:
        self.log_progress('train',epoch * len(self.train_loader) + i)
        if i % opts.print_freq == 0:
          self.display_progress(i,epoch)

  def validate(self,gpu,epoch):
    print("Validating")
    self.create_meters(self.val_loader,'Val: ')

    self.model.eval()

    with torch.no_grad():
      start = time.time()
      #for i, (images, target, geo, date) in enumerate(self.val_loader):
      for i, (images, target) in enumerate(self.val_loader):
          
        images = images.cuda(non_blocking=True)
        # geo = geo.cuda(non_blocking=True)
        # date = date.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        #output = self.model(images,geo,date)
        #output = self.model(images,date)
        output = self.model(images)
        loss = self.criterion(output, target)

        self.update_meters(start, output, target, loss, images)

        if gpu == 0:
          self.log_progress('validation',epoch * len(self.val_loader) + i)
          if i % opts.print_freq == 0:
              self.display_progress(i,epoch)

  def create_meters(self,loader,prefix):
    self.meters = {
      'batch_time':AverageMeter('Time', ':6.3f'),
      'losses':AverageMeter('Loss', ':.4e'),
      'top1':AverageMeter('Acc@1', ':6.2f'),
      'top2':AverageMeter('Acc@2', ':6.2f')
    }

    self.meters['progress'] = ProgressMeter(
      len(loader),
      [self.meters['batch_time'], self.meters['losses'], self.meters['top1'], self.meters['top2']],
      prefix=prefix
    )

  def update_meters(self, start, output, target, loss, batch):
    acc1, acc2 = accuracy(output, target, topk=(1, 2))
    self.meters['losses'].update(loss.item(), batch.size(0))
    self.meters['top1'].update(acc1[0], batch.size(0))
    self.meters['top2'].update(acc2[0], batch.size(0))
    self.meters['batch_time'].update(time.time() - start)

  def display_progress(self,i,epoch):
    self.meters['progress'].display(i)

  def log_progress(self,type,step):
    prefix = "Train: " if type == 'train' else "Validation: "
    if type == 'train':
      self.writer.add_scalar(prefix + 'LR',
        self.scheduler.get_last_lr()[0],
        step)

    self.writer.add_scalar(prefix + 'Acc@1',
      self.meters['top1'].avg,
      step)

    self.writer.add_scalar(prefix + 'Acc@2',
      self.meters['top2'].avg,
      step)

  def save_checkpoint(self,state, model, filename='checkpoint.pth.tar', gpu=0):

    if gpu == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(state, filename)

