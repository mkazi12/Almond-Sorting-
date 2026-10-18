import sys

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINOS Almond Detection")
        self.resize(800, 600)

        self.image_label = QLabel("Choose an image to begin")
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

        image = QPixmap(path)
        if image.isNull():
            self.image_label.setText("Could not open that image.")
            return

        self.image_label.setPixmap(image.scaledToWidth(600))


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())