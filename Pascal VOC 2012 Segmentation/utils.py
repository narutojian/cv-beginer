import torch
from torch import nn
import os
import torchvision
import numpy as np

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def read_voc_images(voc_dir, is_train=True):
    '''
        read pascal voc 2012 image and mask
    '''
    txt_name = os.path.join(voc_dir,'ImageSets', 'Segmentation',
                            'train.txt' if is_train else 'val.txt')
    # print(txt_name)
    mode = torchvision.io.ImageReadMode.RGB
    with open(txt_name,'r') as f:
        images = f.read().split()
    features,labels = [],[]
    for i,fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir,'JPEGImages',f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir,'SegmentationClass',f'{fname}.png'),mode))
    return features,labels

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    '''用数组的下表索引来定位RGB，数组的值为类别'''
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    '''输入一个3通道的image 返回单通道的image 且image的像素值表示类别'''
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# custom dataset
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
    
# train process
def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        # Compute prediction error
        pred = model(X)
        # print(pred)
        loss = loss_fn(pred, y).sum()
        # print(loss)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, device,model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).sum().item()
            pred = nn.Softmax(dim=1)(pred)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # print(y.shape)
    correct /= size*y.shape[1]*y.shape[2]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    
# predict one image
def predict(img,dataloader,device,model):
    X = dataloader.dataset.normalize_image(img).unsqueeze(0)
    pred = model(X.to(device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred,device):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]

# init_weights(m):
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

# one hot encoding 
# 给定一个inputs 输出one hot 形式的outputs 新增的维度在最后一个
def one_hot(x,classes):
    a = torch.eye(classes)
    return a[x]

# iou score for semantic segmentation
def mean_iou(predict, y_true,classes,ignore_backgroud=True,smooth = 1e-6):
    ''' 
        predict shape: N*H*W ; each pixel represent a class 
        y_true: N*H*W; each pixel represent a class 
    '''
    pred_one_hot = one_hot(predict,classes)
    y_true_one_hot = one_hot(y_true,classes)
    axes = [1,2]
    
    intersection = torch.sum(torch.logical_and(pred_one_hot,y_true_one_hot),axes)
    union = torch.sum(torch.logical_or(pred_one_hot,y_true_one_hot),axes)
    if ignore_backgroud:
        intersection = intersection[:,1:]
        union = union[:,1:]
    iou = (intersection+smooth)/(union+smooth)
    return torch.mean(iou,dim=1)
    # return torch.mean(iou)

def iou(predict, y_true,classes,ignore_background=True,smooth = 1e-4):
    ''' 
        predict shape: N*H*W ; each pixel represent a class 
        y_true: N*H*W; each pixel represent a class 
    '''
    iou = np.zeros(classes)
    idx = 0
    if ignore_background:
        idx = 1
    for i in range(idx,classes):
        intersect = torch.sum(torch.logical_and(predict==i,y_true==i))
        union = torch.sum(torch.logical_or(predict==i,y_true==i))
        iou[i] = (intersect+smooth)/(union+smooth)
    return iou