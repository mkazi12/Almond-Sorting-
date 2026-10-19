import sys
import cv2

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.image_loader import ImageLoadError, load_image
from app.inference import run_inference_and_overlay

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINOS Almond Detection")
        self.resize(800, 600)

        self.image_label = QLabel("Choose an image to begin")
        self.image_label.setWordWrap(True)
        open_button = QPushButton("Open almond image")
        open_button.clicked.connect(self.open_image)

        layout = QVBoxLayout()
        layout.addWidget(open_button)
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            "",
            "Images (*.jpg *.jpeg *.png)",
        )

        if not path:
            return

        # mock inference testing
        try:
            # validate  image with image loader
            _ = load_image(path)
            
            # run inference and draw bounding boxes using OpenCV
            result_cv_image = run_inference_and_overlay(path)
            
            # convert OpenCV BGR array to PySide6 QImage (RGB)
            rgb_image = cv2.cvtColor(result_cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # display the rendered image in the UI
            self.image_label.setPixmap(QPixmap.fromImage(q_img).scaledToWidth(600))
            
        except ImageLoadError as error:
            self.image_label.setText(str(error))
        except Exception as e:
            self.image_label.setText(f"Inference Error: {e}")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())