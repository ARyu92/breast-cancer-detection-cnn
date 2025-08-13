# ui_only_app.py
# pip install PyQt5

import sys
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy
)


class ImagePanel(QFrame):
    """Simple titled image panel that scales its pixmap to fit."""
    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title = QLabel(f"<b>{title}</b>")
        self.title.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(QSize(240, 240))
        self.image_label.setStyleSheet("background:#fafafa; border:1px solid #e5e5e5;")

        layout.addWidget(self.title)
        layout.addWidget(self.image_label)

        # start with a neutral placeholder
        self.set_placeholder()

    def set_placeholder(self, size: QSize = QSize(400, 400)):
        pix = QPixmap(size)
        pix.fill(Qt.lightGray)
        self.set_image(pix)

    def set_image(self, pix: QPixmap):
        if pix is None:
            self.image_label.clear()
            return
        self._orig = pix
        self._apply_scaled()

    def resizeEvent(self, event):
        self._apply_scaled()
        super().resizeEvent(event)

    def _apply_scaled(self):
        pix = getattr(self, "_orig", None)
        if pix:
            self.image_label.setPixmap(
                pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breast Cancer Detection")
        self.setMinimumSize(1100, 700)

        # --------- Root layout ----------
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 8, 16, 8)
        root_layout.setSpacing(12)

        # --------- Header ----------
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("<h2>Breast Cancer Detection</h2>")
        subtitle = QLabel("An AI Powered Application")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        # --------- Toolbar ----------
        toolbar = QWidget()
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(0, 0, 0, 0)
        tb_layout.setSpacing(8)

        tb_layout.addStretch(3)   # pad left (mimics your columns proportions)

        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_model = QPushButton("Load Model")

        # hook up no-op handlers (fill these later)
        self.btn_load_image.clicked.connect(self.on_load_image_clicked)
        self.btn_load_model.clicked.connect(self.on_load_model_clicked)

        tb_layout.addWidget(self.btn_load_image)
        tb_layout.addWidget(self.btn_load_model)
        tb_layout.addStretch(10)  # pad right
        root_layout.addWidget(toolbar)

        # --------- Main Body (Patient | CC | MLO) ----------
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(16)

        # Left: Patient info
        info_col = QWidget()
        info_layout = QVBoxLayout(info_col)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(8)

        info_title = QLabel("<b>Patient:</b>")
        self.lbl_patient_name = QLabel("Patient Name: example")
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.lbl_patient_name)
        info_layout.addStretch(1)

        # Center-left: CC image
        self.panel_cc = ImagePanel("Craniocaudal")

        # Center-right: MLO image
        self.panel_mlo = ImagePanel("Mediolateral Oblique")

        # Add with rough weights similar to Streamlit columns [1.5, 3.5, 3.5]
        body_layout.addWidget(info_col, 2)
        body_layout.addWidget(self.panel_cc, 5)
        body_layout.addWidget(self.panel_mlo, 5)

        root_layout.addWidget(body, 1)

        # --------- Footer (Model Options | Console | Markers) ----------
        footer = QWidget()
        foot_layout = QHBoxLayout(footer)
        foot_layout.setContentsMargins(0, 0, 0, 0)
        foot_layout.setSpacing(12)

        # Model Options
        model_opts = QWidget()
        mo_layout = QVBoxLayout(model_opts)
        mo_layout.setContentsMargins(0, 0, 0, 0)
        mo_layout.setSpacing(8)
        mo_layout.addWidget(QLabel("<b>Model Options</b>"))
        self.btn_gear = QPushButton("⚙️")
        mo_layout.addWidget(self.btn_gear)
        mo_layout.addStretch(1)

        # Console
        console_wrap = QWidget()
        console_layout = QVBoxLayout(console_wrap)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(8)
        console_layout.addWidget(QLabel("<b>Console</b>"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        console_layout.addWidget(self.console)

        # Markers
        markers = QWidget()
        mk_layout = QVBoxLayout(markers)
        mk_layout.setContentsMargins(0, 0, 0, 0)
        mk_layout.setSpacing(8)
        self.btn_malignant = QPushButton("Malignant")
        self.btn_benign = QPushButton("Benign")
        self.btn_malignant.clicked.connect(lambda: self.log("Marker: Malignant"))
        self.btn_benign.clicked.connect(lambda: self.log("Marker: Benign"))
        mk_layout.addStretch(1)
        mk_layout.addWidget(self.btn_malignant)
        mk_layout.addWidget(self.btn_benign)
        mk_layout.addStretch(1)

        foot_layout.addWidget(model_opts, 2)
        foot_layout.addWidget(console_wrap, 8)
        foot_layout.addWidget(markers, 2)

        root_layout.addWidget(footer)

        self.setCentralWidget(root)

    # --------- Placeholder handlers (fill in later) ---------
    def on_load_image_clicked(self):
        self.log("Load Image clicked")

    def on_load_model_clicked(self):
        self.log("Load Model clicked")

    def log(self, msg: str):
        self.console.append(msg)


def main():
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("Breast Cancer Detection")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
