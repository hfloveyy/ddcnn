


import torch
import torch.nn as nn
from torch.autograd import Variable
import json
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image

import numpy as np

EPOCH = 100

data_folder = './dataset/images'
data_json = json.load(open('./dataset/data.json'))

image_roots = [os.path.join(data_folder,image_file) \
                for image_file in os.listdir(data_folder)]

#print(image_roots)


class DingDingDataset(Dataset):
    def __init__(self,transform = None):
        self.image_roots = image_roots
        self.yes_or_no = data_json
        self.transform = transform

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image_name = image_root.split("\\")[-1]
        #print(image_name)
        image = Image.open(image_root)
        image = image.convert('RGB')
        image = image.resize((224,224), resample=Image.LANCZOS)
        #image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        flag = int(image_name[-5])
        return image,flag

def data_loader():
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    dataset = DingDingDataset(transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer6 = nn.Linear(1600,10)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.layer6(out)
        return out # 64





if __name__ == '__main__':
    print("init net")
    net = CNNEncoder()



    print("init dataset")
    data = data_loader()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print('training....')
    for epoch in range(EPOCH):
        for i,(image,flag) in enumerate(data):
            image = Variable(image)
            flag = Variable(flag)
            predict_value = net(image)
            loss = criterion(predict_value,flag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_y = torch.max(predict_value, 1)[1].data.numpy().squeeze()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| pred_y:' , pred_y)
            if (epoch + 1) % 5 == 0 and i == 0:
                torch.save(net.state_dict(), "./model.pkl")  # current is model.pkl
                print("save model")






