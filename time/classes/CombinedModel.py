import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as models

from torch.cuda.amp import autocast, GradScaler

import timm

class CombinedModel(nn.Module):
  #def __init__(self,gpu,arch,in_geo_ftrs,in_date_ftrs,out_classes):
  def __init__(self,gpu,arch,out_classes):
    super().__init__()
    
    print("Initializing image model")
    if arch == 'efficientnet':
      model_img = timm.create_model('tf_efficientnet_l2_ns', pretrained=False)
    else:
      model_img = getattr(models,arch)(weights=None,progress=False)

    model_img.classifier.out_features = out_classes
    #num_ftrs = model_img.classifier.in_features
    #model_img.classifier = nn.Linear(num_ftrs,out_classes)
    model_img.cuda(gpu)

    #combo_ftrs = num_ftrs + in_geo_ftrs + in_date_ftrs
    #combo_ftrs = num_ftrs + in_date_ftrs
    #combo_ftrs = out_classes * 2
    #combo_ftrs = out_classes

    #self.hidden = nn.Linear(combo_ftrs, combo_ftrs)
    #self.drop = nn.Dropout(0.5)
    #self.relu = nn.ReLU()
    #self.classifier = nn.Linear(combo_ftrs, out_classes)

    # print("Initializing geo model")
    ### Normalized lat/lng
    #model_geo = nn.Sequential(
    #  nn.Linear(2,num_ftrs)
    #)

    ### Clustered geo one-hot tensor
    # model_geo = nn.Sequential(
    #   nn.Linear(in_geo_ftrs,in_geo_ftrs),
    #   nn.Linear(in_geo_ftrs,out_classes)
    # )
    # model_geo.cuda(gpu)

    # print("Initializing date model")
    ### Single timestamp
    #model_date = nn.Sequential(
    #  nn.Linear(1,num_ftrs)
    #)

    ### Seasonal one-hot tensor
    # model_date = nn.Sequential(
    #   nn.Linear(in_date_ftrs,in_date_ftrs),
    #   nn.Linear(in_date_ftrs,out_classes)
    # )
    # model_date.cuda(gpu)

    self.models = {
      'image':model_img,
      # 'geo':model_geo,
      # 'date':model_date
    }

  #def forward(self, img, date):#geo, date):
  def forward(self, img):

    img = self.models['image'](img)
    #geo = self.models['geo'](geo)
    #date = self.models['date'](date)

    #x = torch.cat((torch.cat((date,geo),dim=1),img),dim=1) #Add geo and date to image
    #x = torch.cat((date,img),dim=1) #Add date to image
    #x = img
    #x = self.hidden(x)
    #x = self.drop(x)
    #x = self.classifier(x) # Classifier num_fts -> out_classes

    return img
