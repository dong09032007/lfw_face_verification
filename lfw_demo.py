import pandas as pd
import os
import torch
print(torch.__version__)
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision import models, transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import time
from tqdm import tqdm, trange
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys 
from vggface2 import Resnet50_ft_dag
from sklearn.metrics import roc_curve, auc


model_type = 'res50' # res50, se50
batch_size = 8
center_crop = 150
dataset_name = 'lfw' #lfw, cplfw

model_checkpoint = 'vggface2/resnet50_ft_dag.pth' 
anno_path = 'lfw/pairs.txt'
image_path = 'lfw/lfw_images'


### model init
net_res50 = Resnet50_ft_dag()
net_res50.load_state_dict(torch.load(model_checkpoint, map_location=torch.device('cpu')))
print(net_res50.meta)

model = net_res50
average_image = torch.tensor(net_res50.meta['mean']).float()
print(average_image)
average_image = average_image.unsqueeze(0).unsqueeze(2).unsqueeze(3)

### move model onto GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = model.to(device)
### set to evaluation mode (for testing)
_ = model.eval()


### read image pair names from annotation file
file = open(anno_path, "r")
lines = file.readlines()
file.close()
if len(lines) == 6001:
    lines = lines[1:]  ## remove header line
test_images = []
test_pairs = []
test_pairs_index = []
for index, line1 in enumerate(tqdm(lines)):
    info1 = line1.replace('\n', '').split('\t')
    if len(info1) == 3:
        name1 = info1[0]
        name2 = info1[0]
        id1 = info1[1].zfill(4)
        id2 = info1[2].zfill(4)
        label = 1
    elif len(info1) == 4:
        name1 = info1[0]
        name2 = info1[2]
        id1 = info1[1].zfill(4)
        id2 = info1[3].zfill(4)
        label = 0
    imname1 = '%s_%s.jpg' % (name1, id1)
    imname2 = '%s_%s.jpg' % (name2, id2)
    impath1 = os.path.join(image_path, name1, imname1)
    impath2 = os.path.join(image_path, name2, imname2)
    assert os.path.exists(impath1) == True
    assert os.path.exists(impath2) == True
    test_images += [impath1, impath2]
    test_pairs.append((impath1, impath2, label))
    test_pairs_index.append((index * 2, index * 2 + 1, label))
    

    
### Make Dataloader    
class BaseDataset(Dataset):
    def __init__(self, image_list, center_crop=150):
        self.n_files = len(image_list)
        
        normalize = transforms.Normalize(mean=[0,0,0], std=[1, 1, 1])
        transf_list = []
        transf_list.extend([
                            transforms.Resize(250),
                            transforms.CenterCrop(center_crop),
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            normalize
        ])
        self.transform = transforms.Compose(transf_list)
        self.image_list = image_list

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    
    def __getitem__(self, idx):        
        img = Image.open(self.image_list[idx])
        return self.transform(self.ensure_3dim(img))
    
    def __len__(self):
        return self.n_files
    
dataset = BaseDataset(test_images, center_crop)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=False)


### extracting features
dim = 2048 ### output of the pretrained model is 2048
data_iterator = tqdm(dataloader)
fea_all = torch.zeros(12000,dim)
for i, x in enumerate(data_iterator):
    x = x * 255 - average_image
    with torch.no_grad():
        _, fea  = model(x.to(device))
    fea = fea.detach().cpu().squeeze()
    ibegin = i*batch_size
    iend = ibegin + x.shape[0]
    fea_all[ibegin:iend] = fea
    
### normalization : length->1   
fea_all_nm = torch.nn.functional.normalize(fea_all, dim=-1)



### calculating distance between positive and negative pairs
P1 = torch.tensor([])
P2 = torch.tensor([])
N1 = torch.tensor([])
N2 = torch.tensor([])

for id1, id2, label in test_pairs_index:
    if label==1:
        P1 = torch.cat((P1, fea_all_nm[id1].unsqueeze(1)),1)
        P2 = torch.cat((P2, fea_all_nm[id2].unsqueeze(1)),1)
    else:
        N1 = torch.cat((N1, fea_all_nm[id1].unsqueeze(1)),1)
        N2 = torch.cat((N2, fea_all_nm[id2].unsqueeze(1)),1)
        
print(P1.shape, P2.shape, N1.shape, N2.shape)
disp = ((P1 - P2)**2).sum(0)
disn = ((N1 - N2)**2).sum(0)



### evaluation with distance threshold = 1
threshold = 1
accp = (disp<=threshold).float().mean()
accn = (disn>threshold).float().mean()
print('mean_disp: %.4f mean_disp: %.4f '%(disp.mean().item(),disn.mean().item()))
print('accp: %.4f accn: %.4f'%(accp.item(), accn.item()))

labels = torch.cat((torch.ones(disp.shape),torch.zeros(disn.shape)))
pred = torch.cat((disp,disn))
pred = pred.max() - pred

labels = labels.tolist()
pred = pred.tolist()
 
### plot ROC curve
fpr, tpr, th = roc_curve(labels, pred , pos_label=1)
plt.plot(fpr,tpr)
print('auc: %.4f'%auc(fpr, tpr))