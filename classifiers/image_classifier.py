import torch
from torchvision import datasets, models, transforms
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os

import torch.nn as nn
import torch.optim as optim

# Define transformations for the training and validation sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet50 model
model_ft = models.resnet50(pretrained=True)

# Modify the final layer to match the number of classes in the dataset
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# Define loss function and optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Training the model
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model

def classify_image(pImage):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert the image to a tensor
    image_tensor = preprocess(pImage).unsqueeze(0).to(device)
    
    # Set the model to evaluation mode
    model_ft.eval()
    
    # Perform the classification
    with torch.no_grad():
        outputs = model_ft(image_tensor)
        _, preds = torch.max(outputs, 1)
    
    # Return the predicted class index
    return preds.item()
    # Define a list of class names (replace with your actual class names)
    class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

    # Return the predicted class name
    return class_names[preds.item()]

if __name__ == '__main__':
    # Train the model
    #model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=25)

    # Save the trained model
    #torch.save(model_ft.state_dict(), 'resnet50_image_classifier.pth')

    # classify shiba image
    # image = cv2.imread(os.path.join(os.path.dirname(__file__), 'testImages/shiba.jpg'))
    # image = cv2.resize(image, (224, 224))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # class_name = classify_image(image)
    # print(class_name)
    image = decode_image(os.path.join(os.path.dirname(__file__), 'testImages/shiba.jpg'))
    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(image).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")