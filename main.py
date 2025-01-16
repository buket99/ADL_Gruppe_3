import sys
import platform
import cv2
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
from classifiers.image_classifier import classify_image, get_available_models
from diffusion_model.diffusion_model import (
    generate_image,
    get_available_diffusion_models,
)
from agents.article_agent import ArticleAgent
from assemblers.article_assembler import ArticleAssembler
import os


class CameraApp(QMainWindow):
    """
    Main application window for capturing and processing images.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Capture")
        self.setGeometry(100, 100, 1200, 700)

        # Define paths using pathlib
        self.base_dir = Path(__file__).resolve().parent
        self.resources_dir = self.base_dir / "resources"
        self.save_directory = self.base_dir / "captured_images"
        self.save_directory.mkdir(parents=True, exist_ok=True)

        # Main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Main Layout
        self.main_layout = QHBoxLayout(self.main_widget)

        # Left Side: Camera Feed and Controls
        self.left_layout = QVBoxLayout()

        # Right Side: Captured Images and Options
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)

        # -- Camera Feed --
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray;")

        # -- Camera Selection --
        self.camera_selector_layout = QHBoxLayout()
        self.camera_selector_label = QLabel("Camera:")
        self.camera_selector_label.setFixedWidth(60)
        self.camera_selector = QComboBox()
        self.camera_selector.setFixedWidth(100)
        self.camera_selector_layout.addWidget(self.camera_selector_label)
        self.camera_selector_layout.addWidget(self.camera_selector)
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        self.populate_cameras()

        # -- Capture Button --
        self.capture_button = QPushButton("Capture")
        self.capture_button.clicked.connect(self.capture_image)

        # -- Feedback Label --
        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet("color: red;")  # Make error messages red

        # -- Assemble Left Layout --
        self.left_layout.addLayout(self.camera_selector_layout)
        self.left_layout.addWidget(self.camera_label)
        self.left_layout.addWidget(self.capture_button)
        self.left_layout.addWidget(self.feedback_label)  # Add feedback label to UI
        self.left_layout.addStretch()

        # -- Captured Images --
        self.captured_images_layout = QHBoxLayout()
        self.captured_images = []
        self.selected_image = None
        self.selected_image_index = -1
        for _ in range(3):
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setFixedSize(150, 150)
            img_label.setStyleSheet("border: 2px solid gray;")
            img_label.mousePressEvent = lambda event, lbl=img_label: self.select_image(
                lbl
            )
            self.captured_images_layout.addWidget(img_label)

        # -- HM Logo --
        self.logo_label = QLabel()
        logo_path = self.resources_dir / "Hochschule_Muenchen_Logo.svg"
        pixmap = QPixmap(str(logo_path)).scaled(
            200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.logo_label.setPixmap(pixmap)

        # -- Model Selection --
        self.model_selection_layout = QVBoxLayout()

        # GPT Model
        self.gpt_label = QLabel("GPT Model:")
        self.gpt_selector = QComboBox()
        self.article_agent = ArticleAgent()
        self.gpt_selector.addItems(self.article_agent.get_available_models())
        self.model_selection_layout.addWidget(self.gpt_label)
        self.model_selection_layout.addWidget(self.gpt_selector)

        # Diffusion Model
        self.diffusion_label = QLabel("Diffusion Model:")
        self.diffusion_selector = QComboBox()
        self.diffusion_selector.addItems(
            get_available_diffusion_models().keys()
        )  # Populate with available models
        self.model_selection_layout.addWidget(self.diffusion_label)
        self.model_selection_layout.addWidget(self.diffusion_selector)

        # Image Classifier
        self.classifier_label = QLabel("Image Classifier:")
        self.classifier_selector = QComboBox()
        self.update_classifier_models()
        self.model_selection_layout.addWidget(self.classifier_label)
        self.model_selection_layout.addWidget(self.classifier_selector)

        # -- Buttons --
        self.buttons_layout = QVBoxLayout()
        self.evaluate_button = QPushButton("Evaluate Image")
        self.evaluate_button.clicked.connect(self.evaluate_selected_image)
        self.generate_article_button = QPushButton("Generate Article")
        self.generate_article_button.clicked.connect(self.generate_article)
        self.open_pdf_button = QPushButton("Open PDF")
        self.open_pdf_button.clicked.connect(self.open_pdf)
        self.buttons_layout.addWidget(self.evaluate_button)
        self.buttons_layout.addWidget(self.generate_article_button)
        self.buttons_layout.addWidget(self.open_pdf_button)

        # -- Assemble Right Layout --
        self.right_layout.addWidget(self.logo_label)
        self.right_layout.addLayout(self.captured_images_layout)
        self.right_layout.addLayout(self.model_selection_layout)
        self.right_layout.addLayout(self.buttons_layout)
        self.right_layout.addStretch()

        # -- Assemble Main Layout --
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # -- Camera Initialization --
        self.camera_index = 0
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Start camera
        self.change_camera()

    def populate_cameras(self):
        """
        Detect available cameras and populate the dropdown.
        """
        available_cameras = []
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap is None or not cap.isOpened():
                    continue
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    continue
                available_cameras.append(f"Camera {i}")
                cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")

        self.camera_selector.clear()
        self.camera_selector.addItems(available_cameras)

        if available_cameras:
            self.camera_index = int(available_cameras[0].split(" ")[-1])
            self.change_camera()
        else:
            self.camera_label.setText("No cameras detected")
            self.feedback_label.setText("No cameras detected")

    def change_camera(self):
        """
        Switch the camera based on the selected index in the dropdown.
        """
        # Release the current camera if it's open
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        try:
            self.camera_index = int(self.camera_selector.currentText().split(" ")[-1])
        except ValueError:
            self.camera_label.setText("Invalid camera index")
            self.feedback_label.setText("Invalid camera index")
            return

        system_name = platform.system()
        if system_name == "Darwin":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        elif system_name == "Windows":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap is None:
            self.camera_label.setText("Camera not found")
            self.feedback_label.setText("Camera not found")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.camera_label.setText(f"Unable to open camera {self.camera_index}")
            self.feedback_label.setText(f"Unable to open camera {self.camera_index}")
            self.cap.release()
            self.cap = None
            return

        if self.cap.isOpened():
            if not hasattr(self, "timer"):
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
        else:
            self.camera_label.setText(f"Unable to open camera {self.camera_index}")
            self.feedback_label.setText(f"Unable to open camera {self.camera_index}")

    def update_frame(self):
        """
        Read frame from camera and display in QLabel.
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(
                    frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_img)
                self.camera_label.setPixmap(
                    pixmap.scaled(
                        self.camera_label.width(),
                        self.camera_label.height(),
                        Qt.KeepAspectRatio,
                    )
                )

    def capture_image(self):
        """
        Capture the current frame, store it, and save to disk.
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if len(self.captured_images) >= 3:
                    self.captured_images.pop(0)

                # Crop to quadratic
                h, w, _ = frame.shape
                if h != w:
                    size = min(h, w)
                    start_x = (w - size) // 2
                    start_y = (h - size) // 2
                    frame = frame[start_y : start_y + size, start_x : start_x + size]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(
                    frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
                )

                self.captured_images.append(qt_img)
                self.update_captured_images()

                # Save the image to disk
                filename = (
                    self.save_directory / f"captured_{len(self.captured_images)}.png"
                )
                cv2.imwrite(str(filename), frame)

                self.feedback_label.setStyleSheet("color: green;")
                self.feedback_label.setText(f"Image saved to:\n{filename}")

    def update_captured_images(self):
        """
        Update the display of captured images.
        """
        for i, img in enumerate(self.captured_images):
            pixmap = QPixmap.fromImage(img).scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.captured_images_layout.itemAt(i).widget().setPixmap(pixmap)

    def select_image(self, label):
        """
        Highlight the selected image with a border.
        """
        # Reset borders
        for i in range(self.captured_images_layout.count()):
            img_label = self.captured_images_layout.itemAt(i).widget()
            img_label.setStyleSheet("border: 2px solid gray;")

        # Find the index of the selected label
        selected_index = -1
        for i in range(self.captured_images_layout.count()):
            if self.captured_images_layout.itemAt(i).widget() == label:
                selected_index = i
                break

        # Set the selected image and highlight
        if selected_index != -1:
            self.selected_image_index = selected_index
            self.selected_image = self.captured_images[selected_index]
            label.setStyleSheet("border: 3px solid blue;")
            print(f"Image {selected_index} selected")

    def closeEvent(self, event):
        """
        Cleanup when the window is closed.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()

    def update_classifier_models(self):
        available_classifier_models = get_available_models()
        self.classifier_selector.clear()

        # Create a dictionary to store model paths
        self.model_paths = {}

        for model_type, model_paths in available_classifier_models.items():
            for model_path in model_paths:
                # Populate the dictionary with model type as key and path as value
                self.model_paths[model_type] = model_path
                # Add only model type to the dropdown
                self.classifier_selector.addItem(model_type)

    def evaluate_selected_image(self):
        if self.selected_image_index >= 0:
            image_path = (
                self.save_directory / f"captured_{self.selected_image_index + 1}.png"
            )

            # Get selected model type from dropdown
            selected_model_type = self.classifier_selector.currentText()

            # Get the path from the dictionary using the selected model type
            path_to_model = self.model_paths.get(selected_model_type)

            if image_path.exists() and path_to_model:
                try:
                    # Classify the image using the selected model type and path
                    self.predicted_class = classify_image(
                        str(image_path), selected_model_type, path_to_model
                    )

                    self.feedback_label.setStyleSheet("color: green;")
                    self.feedback_label.setText(
                        f"Image classified as: {self.predicted_class}"
                    )

                except Exception as e:
                    self.feedback_label.setStyleSheet("color: red;")
                    self.feedback_label.setText(f"Error during classification: {e}")
            elif not path_to_model:
                self.feedback_label.setStyleSheet("color: red;")
                self.feedback_label.setText(
                    "Model path not found for the selected model type."
                )
            else:
                self.feedback_label.setStyleSheet("color: red;")
                self.feedback_label.setText("Image file not found.")
        else:
            self.feedback_label.setStyleSheet("color: red;")
            self.feedback_label.setText("No image selected for evaluation.")

    def generate_article(self):
        """
        Generates the article using the ArticleAgent and passes data to the ArticleAssembler.
        """
        if hasattr(self, "predicted_class"):
            selected_gpt_model = self.gpt_selector.currentText()
            selected_diffusion_model = self.diffusion_selector.currentText()

            # Initialize ArticleAgent with the selected model
            self.article_agent = ArticleAgent(model=selected_gpt_model)

            # Generate article content and image paths
            paragraphs, image_paths = self.article_agent.generate_article_content(
                self.predicted_class, selected_diffusion_model
            )

            # Pass data to ArticleAssembler
            assembler = ArticleAssembler()
            assembler.assemble_article(self.predicted_class, paragraphs, image_paths)

            self.feedback_label.setStyleSheet("color: green;")
            self.feedback_label.setText("Article generated and passed to assembler.")
        else:
            self.feedback_label.setStyleSheet("color: red;")
            self.feedback_label.setText("No image classification available.")

    def open_pdf(self):
        """
        Opens the generated PDF using the default PDF viewer.
        """
        pdf_path = self.base_dir / "output.pdf"
        if pdf_path.exists():
            try:
                if platform.system() == "Windows":
                    os.startfile(pdf_path)
                elif platform.system() == "Darwin":  # macOS
                    os.system(f"open '{pdf_path}'")
                else:  # Linux
                    os.system(f"xdg-open '{pdf_path}'")
                self.feedback_label.setStyleSheet("color: green;")
                self.feedback_label.setText("PDF opened successfully.")
            except Exception as e:
                self.feedback_label.setStyleSheet("color: red;")
                self.feedback_label.setText(f"Error opening PDF: {e}")
        else:
            self.feedback_label.setStyleSheet("color: red;")
            self.feedback_label.setText(
                "PDF file not found. Generate the article first."
            )


def main():
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
