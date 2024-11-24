import torch
from torchvision import datasets, models, transforms
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import urllib
import torch.nn as nn
import torch.optim as optim
from PIL import Image

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


def classify_image(pImage):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert the image to a tensor
    input_tensor = preprocess(pImage)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    
    # Perform the classification
    with torch.no_grad():
        outputs = model(input_batch)    
    
    print(outputs[0])
    

if __name__ == '__main__':
    image = Image.open('dog.jpg')
    print(classify_image(image))
    probabilities = torch.nn.functional.softmax(classify_image(image), dim=0)

        # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())