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
    def init(self):
        super().init()
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
