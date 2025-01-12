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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from urllib.parse import unquote
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import urllib.request
import json
import re
import io
import numpy as np
from pathlib import Path


def scrape_class(class_name, num_images, subset_folder, search_term=None):
    """
    Scrapes images from Google Images by hovering over thumbnails, extracting full-resolution image URLs,
    and downloading images in various formats.
    """
    search_term = search_term if search_term else class_name

    # Zielordner erstellen
    class_folder = os.path.join(subset_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # WebDriver-Einstellungen
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)

    search_url = f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch"
    driver.get(search_url)
    time.sleep(2)

    image_urls = []
    count = 0

    while count < num_images:
        try:
            # Thumbnails abrufen
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
            print(f"[INFO] Found {len(thumbnails)} thumbnails. Attempting to hover...")

            for thumbnail in thumbnails:
                if count >= num_images:
                    break
                try:
                    # Hover über das Thumbnail
                    actions = ActionChains(driver)
                    actions.move_to_element(thumbnail).perform()
                    # time.sleep(0.5)

                    # Suche das `<a>`-Element mit `imgurl=`
                    link_elements = driver.find_elements(
                        By.CSS_SELECTOR, "a[href*='imgurl=']"
                    )
                    for link in link_elements:
                        href = link.get_attribute("href")
                        if "imgurl=" in href:
                            raw_img_url = href.split("imgurl=")[1].split("&")[0]
                            img_url = unquote(raw_img_url)  # Dekodieren der URL

                            # Überprüfen des Bildformats
                            if img_url.lower().endswith(
                                (
                                    ".jpg",
                                    ".jpeg",
                                    ".png",
                                    ".gif",
                                    ".bmp",
                                    ".tiff",
                                    ".webp",
                                )
                            ):
                                if img_url not in image_urls:
                                    image_urls.append(img_url)
                                    print(f"[INFO] Found image URL: {img_url}")
                                    count += 1
                                    if count >= num_images:
                                        break
                except Exception as e:
                    print(
                        f"[WARNING] Could not hover thumbnail or extract image URL: {e}"
                    )
                    continue

            # Scrollen, um mehr Thumbnails zu laden
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] General error during scraping: {e}")
            break

    driver.quit()

    # Bilder herunterladen
    for i, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url, stream=True, timeout=5)
            if response.status_code == 200:
                with Image.open(io.BytesIO(response.content)) as img:
                    # Extrahiere das Format aus der URL oder verwende das PIL-Format
                    ext = os.path.splitext(img_url)[-1].lower()
                    if ext not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".bmp",
                        ".tiff",
                        ".webp",
                    ]:
                        ext = f".{img.format.lower()}"
                    filename = os.path.join(class_folder, f"{class_name}_{i + 1}{ext}")

                    img.save(filename)
                    print(f"[INFO] Saved image {i + 1}/{num_images}: {filename}")
            else:
                print(
                    f"[WARNING] Could not download image {img_url} (status code: {response.status_code})"
                )
        except Exception as e:
            print(f"[ERROR] Failed to download image {img_url}: {e}")

    print(
        f"[INFO] Downloaded {len(image_urls)} images for '{class_name}' to {class_folder}."
    )


def train_model(model_name="alexnet"):
    # Kaggle dataset download
    # dataset_path = kagglehub.dataset_download("vencerlanz09/bottle-synthetic-images-dataset")
    # print("Dataset downloaded to:", dataset_path)
    dataset_path = (
        "C:\\Studium\\Master\\WiSe-24-25\\Vorlesungen\\advanced deep learning"
    )

    # # Dataset preparation
    data_dir = f"{dataset_path}/Bottle Images Subset"  # Root directory for the dataset
    # target_dir = f"{dataset_path}/Bottle Images/Bottle Images Subset"  # Directory for the subset
    # os.makedirs(target_dir, exist_ok=True)

    # # Ensure only 100 files per folder
    # for folder in os.listdir(data_dir):
    #     folder_path = os.path.join(data_dir, folder)
    #     if os.path.isdir(folder_path):
    #         subset_folder_path = os.path.join(target_dir, folder)
    #         os.makedirs(subset_folder_path, exist_ok=True)

    #         # Get the list of files and sort them
    #         files = sorted(os.listdir(folder_path))[:150]  # Take only the first 100 files
    #         for file in files:
    #             src = os.path.join(folder_path, file)
    #             dest = os.path.join(subset_folder_path, file)
    #             shutil.copy(src, dest)  # Copy file to the new subset folder

    # print(f"Subset created in {target_dir}")

    # Scrape additional images for a class (example)
    # scrape_class("Orange Juice", 150, target_dir, "Orange Juice Drink")
    # scrape_class("Coffee", 150, target_dir, "coffee drink photo")
    # scrape_class("Milk", 150, target_dir, "milk")
    # scrape_class("Tea", 150, target_dir, "tea drink")
    # scrape_class("Coke", 150, target_dir, "cola limonade flasche")
    # scrape_class("Orange Soda", 150, target_dir, "orange soda drink")
    # scrape_class("Energy Drink", 150, data_dir, "energy drink")

    # Transforms for the dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Common input size
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Train-Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # PyTorch Lightning module
    class FineTunedModel(pl.LightningModule):
        def __init__(self, model_name, num_classes):
            super(FineTunedModel, self).__init__()
            self.model_name = model_name
            if model_name == "alexnet":
                self.model = torch.hub.load(
                    "pytorch/vision:v0.10.0", "alexnet", pretrained=True
                )
                self.model.classifier[6] = nn.Linear(4096, num_classes)
            elif model_name == "resnet50":
                self.model = torch.hub.load(
                    "pytorch/vision:v0.10.0", "resnet50", pretrained=True
                )
                self.model.fc = nn.Linear(2048, num_classes)
            elif model_name == "vit":
                self.model = models.vit_b_16(pretrained=True)
                self.model.heads.head = nn.Linear(
                    self.model.heads.head.in_features, num_classes
                )
            else:
                raise ValueError(
                    "Unsupported model. Choose either 'alexnet', 'resnet50', or 'vit'."
                )
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.model.parameters(), lr=1e-4)

    # Model training
    model = FineTunedModel(model_name, num_classes=len(dataset.classes))

    # Add checkpointing callback
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    # Measure training time
    start_time = time.time()

    trainer = pl.Trainer(
        max_epochs=50,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()

    # Calculate total training time
    training_time = end_time - start_time

    # Retrieve the best validation accuracy
    val_acc = trainer.callback_metrics.get("val_acc", None)
    val_acc_value = val_acc.item() if val_acc is not None else "N/A"

    # Save the trained model
    model_save_path = os.path.join(
        os.path.dirname(__file__),
        f"classifier/model/bottle_fine_tuned_{model_name}.pth",
    )
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
    def evaluate_and_plot_confusion_matrix(
        model, test_loader, classes, model_name="default_model"
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                print(
                    f"[DEBUG] Class {class_name}: Correct={class_correct}, Total={class_total}, Accuracy={class_accuracy:.4f}"
                )

        print(f"[INFO] Confusion matrix and accuracy logged successfully.")

    # Test set evaluation
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    evaluate_and_plot_confusion_matrix(model.model, test_loader, dataset.classes)


class_names = [
    "Beer Bottles",
    "Coffee",
    "Coke",
    "Drinking Bottle",
    "Energy Drink",
    "Milk",
    "Orange Juice",
    "Orange Soda",
    "Tea",
    "Water",
    "Wine",
]


def load_model(model_type, model_path):
    """
    Load the specified model (AlexNet, ResNet50, or Vision Transformer) with custom weights.
    """
    if model_type == "alexnet":
        model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=False)
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, len(class_names)
        )
    elif model_type == "resnet50":
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    elif model_type == "vit":
        model = torch.hub.load("pytorch/vision:v0.10.0", "vit_b_16", pretrained=False)
        model.heads.head = torch.nn.Linear(
            model.heads.head.in_features, len(class_names)
        )
    else:
        raise ValueError(
            "Unsupported model type. Choose from 'alexnet', 'resnet50', or 'vit'."
        )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def classify_image(image_path, model_type="alexnet", model_path="model.pth"):
    """
    Classify a single image using the specified model.
    """
    # Load the specified model
    model = load_model(model_type, model_path)

    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Return the class name
    predicted_class = class_names[predicted.item()]
    return predicted_class


def classify_images_in_directory(
    directory_path, model_type="alexnet", model_path="model.pth"
):
    """
    Classify all images in a directory using the specified model.
    """
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
        ):
            image_path = os.path.join(directory_path, filename)
            prediction = classify_image(image_path, model_type, model_path)
            print(f"Image: {filename}, Prediction: {prediction}")


def preprocess_image(image_path):
    """
    Preprocess the input image for the model.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image


def generate_saliency_map(model, image_path, class_names):
    """
    Generate and visualize the saliency map for a given image and model.
    """
    # Preprocess the image
    input_image = preprocess_image(image_path)
    input_image.requires_grad_()  # Enable gradients for the input image

    # Forward pass
    model.eval()
    output = model(input_image)
    predicted_class = output.argmax(dim=1).item()

    print(f"[INFO] Predicted class: {class_names[predicted_class]}")

    # Backward pass
    model.zero_grad()
    output[0, predicted_class].backward()  # Compute gradients for the predicted class

    # Get the saliency map
    saliency = input_image.grad.abs().squeeze().max(dim=0)[0].detach().numpy()

    # Visualize the saliency map as an overlay
    original_image = Image.open(image_path).resize((224, 224))
    original_image_np = np.array(original_image)

    # Normalize the saliency map to [0, 255]
    saliency_normalized = (saliency - saliency.min()) / (
        saliency.max() - saliency.min()
    )
    saliency_normalized = (saliency_normalized * 255).astype(np.uint8)

    # Create a colormap for the saliency map
    colormap = plt.cm.jet(saliency_normalized / 255.0)
    saliency_colored = (colormap[:, :, :3] * 255).astype(np.uint8)

    # Overlay the saliency map on the original image
    overlay = 0.5 * original_image_np + 0.5 * saliency_colored
    overlay = overlay.astype(np.uint8)

    # Display the images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(saliency_normalized, cmap="hot")
    plt.axis("off")
    plt.title("Saliency Map")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Overlay")

    plt.show()


def classify_and_visualize_with_saliency(
    model_path, image_path, model_type, class_names
):
    """
    Classify an image and generate the saliency map.
    """
    # Load the trained model
    model = load_model(model_type, model_path)

    # Generate the saliency map
    generate_saliency_map(model, image_path, class_names)


if __name__ == "__main__":
    # Specify the directory containing the test images
    # test_images_directory = os.path.join(os.getcwd(), "classifiers/testImages")
    # classify_images_in_directory(test_images_directory)

    # model_path = "bottle_fine_tuned_alexnet.pth"
    # image_path = "classifiers/testImages/KartonMilch2.jpg"

    # classify_and_visualize_with_saliency(model_path, image_path, class_names)
    # train_model("alexnet")

    # Specify the directory containing the test images
    test_image_name = "input_20250111_184715.jpg"
    BASE_DIR = (
        Path(__file__).resolve().parent
    )  # path to the folder of image_classifier.py
    test_images_directory = BASE_DIR.parent / "classifiers" / "input"

    model_path = os.path.join(
        os.path.dirname(__file__), "model/bottle_fine_tuned_alexnet.pth"
    )

    # Classify and visualize saliency for each image in the directory
    for filename in os.listdir(test_images_directory):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
        ):
            image_path = os.path.join(test_images_directory, filename)
            print(f"Processing image: {filename}")
            classify_and_visualize_with_saliency(
                model_path, image_path, model_type="alexnet", class_names=class_names
            )
