import sys
import cv2

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.image_loader import ImageLoadError, load_image
from app.inference import run_inference_and_overlay
from app.recorded_setup import SAMPLE_MEDIA_PATH, setup_text


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINOS Almond Detection")
        self.resize(800, 600)

        self.recorded_setup_label = QLabel(setup_text())
        self.recorded_setup_label.setObjectName("recorded_setup")
        self.recorded_setup_label.setWordWrap(True)

        self.image_label = QLabel("Choose an image to begin")
        self.image_label.setObjectName("image_label")
        self.image_label.setWordWrap(True)

        open_button = QPushButton("Open almond image")
        open_button.clicked.connect(self.open_image)
        sample_button = QPushButton("Load sample footage")
        sample_button.setObjectName("load_sample_button")
        sample_button.clicked.connect(self.load_sample_footage)

        buttons = QHBoxLayout()
        buttons.addWidget(open_button)
        buttons.addWidget(sample_button)

        layout = QVBoxLayout()
        layout.addWidget(self.recorded_setup_label)
        layout.addLayout(buttons)
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
        self.display_image_path(path)

    def load_sample_footage(self):
        self.display_image_path(str(SAMPLE_MEDIA_PATH))

    def display_image_path(self, path: str):
        try:
            _ = load_image(path)
            result_cv_image = run_inference_and_overlay(path)
            rgb_image = cv2.cvtColor(result_cv_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            q_img = QImage(
                rgb_image.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888,
            )
            self.image_label.setPixmap(QPixmap.fromImage(q_img).scaledToWidth(600))
        except ImageLoadError as error:
            self.image_label.setText(str(error))
        except Exception as error:
            self.image_label.setText(f"Inference Error: {error}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
