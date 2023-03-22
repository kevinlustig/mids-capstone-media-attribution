from datetime import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from pydantic import BaseModel,Extra
# import joblib
# from sqlalchemy.orm import Session
import os
import boto3
import paho.mqtt.client as mqtt
import cv2
import requests
import pickle
import sklearn

###############################################################
ACCESS_KEY_ID = 'AKIAV2JZKDCAVRPF4HFP'
SECRET_ACCESS_KEY = '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'
BUCKET = 'w251-range-project-bucket'
###############################################################
# LOCAL_REDIS_URL = "redis://redis.w255.svc.cluster.local:6379"



# with open('geo_model.pkl', 'rb') as f:
#   classifier_geo = pickle.load(f)
#   print("Read geo_model")
#   geo_classes = classifier_geo.weights_.shape[0]

def geo_transform(geo):
  ### Normalized timestamp approach
  #return torch.FloatTensor([
  #  (float(geo[0]) - self.geo_mean[0]) / self.geo_std[0],
  #  (float(geo[1]) - self.geo_mean[1]) / self.geo_std[1]
  #])
  geo = [1,0,0,0,0,0,0,0,0,0]
  ### Clustering approach
  # geo = torch.LongTensor(classifier_geo.predict([geo]))
  # geo = F.one_hot(geo,num_classes=geo_classes)
  print("Get geo info")
  return geo

def date_to_season_one_hot(d):
  #One-hot encode from a date based on Kenya's wet/dry seasons 
  month = int(datetime.strptime(d, '%Y%m%d').strftime('%-m'))
  return np.asarray([
    1 if month > 11 or month <= 3 else 0, #winter dry season
    1 if month > 3 and month <= 6 else 0, #spring wet season
    1 if month > 6 and month <= 10 else 0, #summer dry season
    1 if month > 10 and month <= 11 else 0, #fall wet season
  ])

def date_transform(d):
  ### Normalized timestamp approach
  #return torch.FloatTensor([
  #  (float(self.date_to_timestamp(d)) - self.date_mean) / self.date_std
  #])

  ### Seasonal approach 
  # return torch.FloatTensor(date_to_season_one_hot(d))
  return date_to_season_one_hot(d)

def get_date():
  d = datetime.now().strftime('%Y%m%d')
  return date_transform(d)

def get_ip():
  response = requests.get('https://api64.ipify.org?format=json').json()
  return response["ip"]

def get_location():
  ip_address = get_ip()
  response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
  location_data = {
      "latitude": response.get("latitude"),
      "longitude": response.get("longitude")
  }
  return location_data

def get_geo():
  geo = [get_location()["latitude"],get_location()["longitude"]]
  geo = geo_transform(geo)
  geo = geo[0:10]
  print("geo",geo)
  return geo

class CombinedModel(nn.Module):
  def __init__(self,gpu,arch,num_classes):
    super().__init__()
    
    print("Initializing image model")
    model_img = getattr(models,arch)(progress=False)
    #Extract number of features from classifier layer
    num_ftrs = model_img.classifier.in_features 
    #Drop classifier layer - we will classify after merging in geo and date
    model_img.classifier = nn.Identity()
    model_img.cuda(gpu)

    print("Initializing geo model")
    ### Normalized lat/lng
    #model_geo = nn.Sequential(
    #  nn.Linear(2,num_ftrs)
    #)

    ### Clustered geo one-hot tensor
    model_geo = nn.Sequential(
     nn.Linear(10,num_ftrs)
    )
    model_geo.cuda(gpu)

    print("Initializing date model")
    ### Single timestamp
    #model_date = nn.Sequential(
    #  nn.Linear(1,num_ftrs)
    #)

    ### Seasonal one-hot tensor
    model_date = nn.Sequential(
     nn.Linear(4,num_ftrs)
    )
    model_date.cuda(gpu)

    self.models = {
      'image':model_img,
      'geo':model_geo,
      'date':model_date
    }
    
    self.hidden = nn.Linear(num_ftrs, num_ftrs)
    self.drop = nn.Dropout(0.5)
    self.relu = nn.ReLU()
    self.classifier = nn.Linear(num_ftrs, num_classes)

  def forward(self, img, geo, date):

    img = self.models['image'](img)
    geo = self.models['geo'](geo)
    date = self.models['date'](date)

    x = torch.add(torch.add(img,geo),date) #Add geo and date to image
    x = self.hidden(x) #Hidden layer/ReLU
    x = self.relu(x)
    x = self.drop(x) #Dropout
    x = self.classifier(x) # Classifier num_fts -> num_classes

    return x

class ModelOptions(object):
    def __init__(self, **opts):
        self.__dict__.update(opts)

opts = ModelOptions(**{
    'start_epoch':0,
    'num_classes':4,
    'epochs':8,
    'lr':.05,
    'momentum':0.9,
    'weight_decay':5e-4,
    'print_freq':50,
    'batch_size':10,
    'workers':0,
    'traindir':"data/train",
    'valdir':"data/val",
    'image_size':224,
    'rgb_mean':[0.47889522, 0.47227842, 0.43047404],
    'rgb_std':[0.229, 0.224, 0.225],
    'checkpoint_path': "./checkpoint.pth.tar",
    #'arch': "efficientnet_v2_s"
    #'arch': "resnext101_64x4d"
    'arch': "densenet201"
})


model = CombinedModel(0,opts.arch,opts.num_classes)
print("Model Initialize")
optimizer = optim.SGD(
        model.parameters(),
        lr=opts.lr,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )


PATH = "checkpoint.pth.tar"
print("Read Model")
checkpoint = torch.load(PATH)
# checkpoint = torch.load(PATH, map_location="cuda:0")
print("checkpoint Load")
model.load_state_dict(checkpoint['state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_dict'])
print("Model Load Complete")

IMG_SIZE = 224
imagenet_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
imagenet_std_RGB = [0.229, 0.224, 0.225]
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB)
    ])

model.cuda()
print("Model Cuda")
model.eval()
print("Model Eval")

def predict(img):
  # frame = read_image('image.jpeg')
  img_normalized = transform(img).float()
  print("Tansform Float")
  img_normalized = img_normalized.unsqueeze_(0)
  print("Unsqueeze")
  # frame = transform.ToPILImage()(img)
  geo = get_geo()
  date = get_date()
  # geo = torch.FloatTensor(np.asarray(geo)).cuda()
  # date = torch.FloatTensor(np.asarray(date)).cuda()
  geo = torch.FloatTensor(np.asarray(geo)).cuda()
  date = torch.FloatTensor(np.asarray(date)).cuda()
  img_normalized = img_normalized.cuda()
  # with torch.no_grad():
  output =model(img_normalized,geo,date)
#     print(output)
  output = output.cpu().data.numpy().argmax()
#     print(index_)
#     # classes = train_ds.classes
#     # class_name = classes[index]
  print(output)
  return output,geo,date
  # return output

# LOCAL_MQTT_HOST = "mosquitto-service"
# LOCAL_MQTT_HOST = "mosquitto-service.default.svc.cluster.local"

LOCAL_MQTT_HOST ="10.43.152.40"
LOCAL_MQTT_PORT = 30001
LOCAL_MQTT_TOPIC ="image"
LOCAL_MQTT_TOPIC_pub = "predict"

print(f'{LOCAL_MQTT_HOST} {LOCAL_MQTT_PORT}')
print(type(LOCAL_MQTT_HOST), type(LOCAL_MQTT_PORT))


#S3 Connection
s3 = boto3.client('s3', 
aws_access_key_id=ACCESS_KEY_ID,
aws_secret_access_key=SECRET_ACCESS_KEY)

def on_connect_local(client, userdata, flags, rc):
  print("connected to local broker with rc: " + str(rc))
  client.subscribe(LOCAL_MQTT_TOPIC)

def on_message(client,userdata, msg):
  try:
    img = msg.payload
    print("Message received.")
    print("topic: ",msg.topic)
    topic_pub = msg.topic

    date_time = datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
    file_name = str(date_time) + '_image.png'
    nparr = np.frombuffer(img, np.uint8)
    print("NPARR")
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("IMDECODE")
  except:
    print("Read Image Failure")
  output,geo,date  = predict(img)
  # output = 1
  # geo = 0
  # date = 0
  print("geo:",geo)
  print("date:",date)
  print("output",output)
  try:
    cv2.imwrite(file_name, img)
    print("imwrite")
  except:
    print("Write Image Failure")
  try:
    broker_sub.publish(LOCAL_MQTT_TOPIC_pub,output)
    print("publish")
  except:
    print("Publish Failure")
  try:
    text_file_name = str(date_time) + '_image.txt'
    text = str(date_time) + "," + str(geo) + "," + str(date) + "," + str(output)
    with open(text_file_name, 'w') as f:
      f.write(text)
      print("write text")
    s3.upload_file(file_name, BUCKET, str(file_name))
    print('wrote file name: ', file_name)
    s3.upload_file(text_file_name, BUCKET, str(text_file_name))
    print('wrote file name: ', text_file_name)
  except:
    print("AWS Upload Failure")

# Initialize the client for subscribing to internal broker
broker_sub = mqtt.Client("brokerSub")
broker_sub.on_connect = on_connect_local
broker_sub.on_message = on_message
# broker_sub.on_publish = on_publish

# Connect to local broker
broker_sub.connect(LOCAL_MQTT_HOST, keepalive=1200)
print("conntect")
broker_sub.loop_forever()




# model = CombinedModel(0,opts.arch,opts.num_classes)

# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# print("Model read finished")

### torch.cuda.set_device(GPU)
# model.cuda()

# model.eval()




# img_normalized = transform(frame).float()
# img_normalized = img_normalized.unsqueeze_(0)
# img_normalized = img_normalized.cuda(non_blocking=True)




    
