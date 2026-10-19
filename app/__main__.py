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

from app.image_loader import ImageLoadError, load_image


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

        try:
            image = load_image(path)
        except ImageLoadError as error:
            self.image_label.setText(str(error))
            return

        # Keep the decoded image at full resolution; scale only the display copy.
        self.image_label.setPixmap(QPixmap.fromImage(image).scaledToWidth(600))


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
