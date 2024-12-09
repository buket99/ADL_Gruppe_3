import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import shutil
import requests
from bs4 import BeautifulSoup

def scrape_class(class_name, num_images, subset_folder, search_term=None):
    """
    Downloads images using Google Image Search and saves them into the subset folder.

    Args:
        class_name (str): Name of the class (used as folder name).
        num_images (int): Number of images to download.
        subset_folder (str): Path to the folder where the scraped images will be saved.
        search_term (str): Custom search term for Google Images (optional).
    """
    # Use class name as search term if none is provided
    search_term = search_term if search_term else class_name

    # Create the target directory for the class
    class_folder = os.path.join(subset_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

    GOOGLE_IMAGE = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'
    URL_input = GOOGLE_IMAGE + 'q=' + search_term
    print(f"[INFO] Fetching images for '{search_term}' from {URL_input}...")
    
    URLdata = requests.get(URL_input)
    soup = BeautifulSoup(URLdata.text, "html.parser")
    ImgTags = soup.find_all('img')
    
    print("[INFO] Downloading images...")
    i = 0  # Image counter

    for link in ImgTags:
        if i >= num_images:
            break
        try:
            # Extract image URL
            images = link.get('src')
            if not images:
                continue
            
            # Determine file extension
            ext = images[images.rindex('.'):]
            if ext.startswith('.png'):
                ext = '.png'
            elif ext.startswith('.jpg'):
                ext = '.jpg'
            elif ext.startswith('.jfif'):
                ext = '.jfif'
            elif ext.startswith('.svg'):
                ext = '.svg'
            else:
                ext = '.jpg'

            # Download and save the image
            response = requests.get(images, stream=True, timeout=5)
            if response.status_code == 200:
                filename = os.path.join(class_folder, f"{class_name}_{i}{ext}")
                with open(filename, 'wb') as file:
                    shutil.copyfileobj(response.raw, file)
                print(f"[DEBUG] Saved image {i + 1}/{num_images}: {filename}")
                i += 1
        except Exception as e:
            print(f"[ERROR] Failed to download an image: {e}")
            continue

    print(f"[INFO] Downloaded {i} images for '{class_name}' to {class_folder}.")
    
def main(model_name="alexnet"):
    # Kaggle dataset download
    dataset_path = kagglehub.dataset_download("vencerlanz09/bottle-synthetic-images-dataset")
    print("Dataset downloaded to:", dataset_path)

    # Dataset preparation
    data_dir = f"{dataset_path}/Bottle Images/Bottle Images"  # Root directory for the dataset
    target_dir = f"{dataset_path}/Bottle Images/Bottle Images Subset"  # Directory for the subset
    os.makedirs(target_dir, exist_ok=True)

    # Ensure only 100 files per folder
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            subset_folder_path = os.path.join(target_dir, folder)
            os.makedirs(subset_folder_path, exist_ok=True)

            # Get the list of files and sort them
            files = sorted(os.listdir(folder_path))[:100]  # Take only the first 100 files
            for file in files:
                src = os.path.join(folder_path, file)
                dest = os.path.join(subset_folder_path, file)
                shutil.copy(src, dest)  # Copy file to the new subset folder

    print(f"Subset created in {target_dir}")

    # Scrape additional images for a class (example)
    scrape_class("Orange Juice", 100, target_dir, "Orange Juice Drink")  # Add 50 images of "Water Bottle"

    # Transforms for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Common input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=target_dir, transform=transform)

    # Train-Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)

    # PyTorch Lightning module
    class FineTunedModel(pl.LightningModule):
        def __init__(self, model_name, num_classes):
            super(FineTunedModel, self).__init__()
            self.model_name = model_name
            if model_name == "alexnet":
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
                self.model.classifier[6] = nn.Linear(4096, num_classes)
            elif model_name == "resnet50":
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                self.model.fc = nn.Linear(2048, num_classes)
            elif model_name == "vit":
                self.model = models.vit_b_16(pretrained=True)
                self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
            else:
                raise ValueError("Unsupported model. Choose either 'alexnet', 'resnet50', or 'vit'.")
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.model.parameters(), lr=1e-4)

    # Model training
    model = FineTunedModel(model_name, num_classes=len(dataset.classes))
    
    # Add checkpointing callback
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)

    # Measure training time
    start_time = time.time()
    
    trainer = pl.Trainer(
        max_epochs=4,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()
    
    # Calculate total training time
    training_time = end_time - start_time
    
    # Retrieve the best validation accuracy
    val_acc = trainer.callback_metrics.get("val_acc", None)
    val_acc_value = val_acc.item() if val_acc is not None else "N/A"

    # Save the trained model
    model_save_path = f"bottle_fine_tuned_{model_name}.pth"
    torch.save(model.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Log results
    log_file = "training_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Validation Accuracy: {val_acc_value}\n")
        f.write(f"Model saved at: {model_save_path}\n")
        f.write("-" * 40 + "\n")

    print(f"Training log saved to {log_file}")

    # Confusion matrix
    def evaluate_and_plot_confusion_matrix(model, test_loader, classes, model_name="default_model"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")

        model = model.to(device)
        model.eval()
        all_preds = []
        all_labels = []

        print("[INFO] Starting evaluation...")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                print(f"[DEBUG] Processing batch {batch_idx + 1}/{len(test_loader)}")
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("[INFO] Evaluation complete. Calculating confusion matrix...")

        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)

        # Uncomment to display the confusion matrix
        # plt.title("Confusion Matrix")
        # plt.show()

        # Calculate overall accuracy
        total_correct = sum(cm[i, i] for i in range(len(classes)))
        total_samples = cm.sum()
        accuracy = total_correct / total_samples
        print(f"[INFO] Overall Accuracy: {accuracy:.4f}")

        # Create logs folder if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"test_dataset_log_{model_name}.txt")

        print(f"[INFO] Writing results to log file: {log_file_path}")
        
        # Write confusion matrix and accuracy to log file
        with open(log_file_path, "w") as log_file:
            log_file.write("Confusion Matrix:\n")
            for row in cm:
                log_file.write(f"{row}\n")
            log_file.write("\n")

            log_file.write(f"Overall Accuracy: {accuracy:.4f}\n\n")

            # Accuracy per class
            log_file.write("Class-wise Accuracy:\n")
            for idx, class_name in enumerate(classes):
                class_correct = cm[idx, idx]
                class_total = cm[idx].sum()
                class_accuracy = class_correct / class_total if class_total > 0 else 0.0
                log_file.write(f"{class_name}: {class_accuracy:.4f}\n")

                # Debugging information for each class
                print(f"[DEBUG] Class {class_name}: Correct={class_correct}, Total={class_total}, Accuracy={class_accuracy:.4f}")

        print(f"[INFO] Confusion matrix and accuracy logged successfully.")
        

    # Test set evaluation
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    evaluate_and_plot_confusion_matrix(model.model, test_loader, dataset.classes)


if __name__ == '__main__':
    # Choose the model: 'alexnet', 'resnet50', or 'vit'
    main(model_name="alexnet")
