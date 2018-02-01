import torch
from main import CNNEncoder
from PIL import Image
from torch.autograd import Variable
import os
import torchvision.transforms as transforms



def preprocess(image):
    w, h = image.size
    top =  (h - w)/2

    image = image.crop((0,top,w,w+top))
    image = image.convert('RGB')
    image = image.resize((224,224), resample=Image.LANCZOS)

    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    image = transform(image)

    return image

def main():

    # init conv net

    net = CNNEncoder()
    if os.path.exists("./model.pkl"):
        net.load_state_dict(torch.load("./model.pkl",map_location=lambda storage, loc: storage))
        print("load model")

    print("load ok")

    image = Image.open('./IMG_32470.JPG')
    image = preprocess(image)

    image = Variable(image.unsqueeze(0))
    predict_value = net(image)
    pred_y = torch.max(predict_value, 1)[1].data.numpy().squeeze()
    print(pred_y)

if __name__ == '__main__':
    main()