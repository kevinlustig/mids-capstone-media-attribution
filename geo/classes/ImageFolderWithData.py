from datetime import datetime
import csv
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets

from sklearn.mixture import GaussianMixture

class ImageFolderWithData(torchvision.datasets.ImageFolder):
  def __init__(self,folder,transform=None,data_file=None):
    super().__init__(folder,transform)

    # if data_file:
    #   with open(data_file, mode='r', encoding='utf-8-sig') as csvfile:
    #     data = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(csvfile, skipinitialspace=True, delimiter=';')]
    #     self.data = {datum["Photo_ID"]:datum for datum in data}

        ### Geo clustering w/ GMM
        # with open('geo_model_gmm.pkl', 'rb') as f:
        #   self.classifier_geo = pickle.load(f)
        #   self.geo_classes = self.classifier_geo.weights_.shape[0]

        ### Geo clustering w/ KMeans - PREFERRED
        # with open('geo_model_kmeans.pkl', 'rb') as f:
        #   self.classifier_geo = pickle.load(f)
        #   self.geo_classes = self.classifier_geo.n_clusters

        ### Geo normalization metrics
        # lngs = np.array([datum['Longitude'] for datum in data]).astype(np.float)
        # lats = np.array([datum['Latitude'] for datum in data]).astype(np.float)

        # self.geo_mean = (np.mean(lats),np.mean(lngs))
        # self.geo_std = (np.std(lats),np.std(lngs))

        ### Date normalization metrics
        # dates = np.array([self.date_to_timestamp(datum['Date']) for datum in data]).astype(np.float)

        # self.date_mean = np.mean(dates)
        # self.date_std = np.std(dates)

        # self.date_classes = 4

  @staticmethod
  def date_to_timestamp(d):
    return datetime.strptime(d, '%Y%m%d').strftime('%s')

  @staticmethod
  def date_to_season_one_hot(d):
    #One-hot encode from a date based on Kenya's wet/dry seasons 
    month = int(datetime.strptime(d, '%Y%m%d').strftime('%-m'))
    return np.asarray([
      1 if month > 11 or month <= 3 else 0, #winter dry season
      1 if month > 3 and month <= 6 else 0, #spring wet season
      1 if month > 6 and month <= 10 else 0, #summer dry season
      1 if month > 10 and month <= 11 else 0, #fall wet season
    ])

  @staticmethod
  def get_geo(g):
    return [float(g["Longitude"]),float(g["Latitude"])]

  def geo_transform(self,geo):
    ### Normalized timestamp approach
    #return torch.FloatTensor([
    #  (float(geo[0]) - self.geo_mean[0]) / self.geo_std[0],
    #  (float(geo[1]) - self.geo_mean[1]) / self.geo_std[1]
    #])

    ### Clustering approach
    geo = torch.LongTensor(self.classifier_geo.predict([geo]))
    geo = F.one_hot(geo,num_classes=self.geo_classes)
    return torch.FloatTensor(np.asarray(geo[0]))

  def date_transform(self,d):
    ### Normalized timestamp approach
    #return torch.FloatTensor([
    #  (float(self.date_to_timestamp(d)) - self.date_mean) / self.date_std
    #])

    ### Seasonal approach 
    return torch.FloatTensor(self.date_to_season_one_hot(d))

  def __getitem__(self,index):
      data, target = super().__getitem__(index)
      #Get image id from path
      # path = self.imgs[index][0]
      # img_id = path.split("/").pop()
      # #Some images are augmented - these have an underscore in the filename; borrow geo/date data from the original image by converting the ID
      # if "_" in img_id:
      #   img_id = img_id.split("_")[0] + ".jpg"
      # datum = self.data[img_id]
      #return data, target, self.geo_transform(self.get_geo(datum)), self.date_transform(datum['Date'])
      return data, target