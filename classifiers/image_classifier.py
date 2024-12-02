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


def main(model_name="alexnet"):
    # Kaggle dataset download
    dataset_path = kagglehub.dataset_download("vencerlanz09/bottle-synthetic-images-dataset")
    print("Dataset downloaded to:", dataset_path)

    # Dataset preparation
    data_dir = f"{dataset_path}/Bottle Images/Bottle Images"  # Root directory for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Common input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

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
    def evaluate_and_plot_confusion_matrix(model, test_loader, classes):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    # Test set evaluation
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    evaluate_and_plot_confusion_matrix(model.model, test_loader, dataset.classes)


if __name__ == '__main__':
    # Choose the model: 'alexnet', 'resnet50', or 'vit'
    main(model_name="vit")
