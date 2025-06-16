import sys
import os
import cv2
import numpy as np
import time
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(cv2.__file__), "qt", "plugins", "platforms")
from PyQt5 import QtCore, QtGui, QtWidgets
from view.face_ui import Ui_Form
from model.siamese_model import load_model, verify
from datetime import datetime


def augment_image(image):
    import tensorflow as tf
    aug = []
    image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    aug.append(tf.image.random_flip_left_right(image_tensor))
    aug.append(tf.image.random_brightness(image_tensor, max_delta=0.1))
    aug.append(tf.image.random_contrast(image_tensor, 0.8, 1.2))
    return [img.numpy() for img in aug]


class FaceVerificationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Inisialisasi webcam
        self.cap = cv2.VideoCapture(4)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)

        # Load model
        self.model = load_model()

        # Connect tombol
        self.ui.verify_button.clicked.connect(self.run_verification)
        self.ui.register_button.clicked.connect(self.run_registration)

        self.user_name = None
        self.frame = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame = frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        qt_pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(self.ui.label_camera.width(), self.ui.label_camera.height())
        self.ui.label_camera.setPixmap(qt_pixmap)

    def log(self, text):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.ui.terminal.append(f"[{timestamp}] {text}")

    def run_verification(self):
        if self.frame is None:
            self.log("Tidak ada frame dari kamera.")
            return
        os.makedirs("application_data/input_image", exist_ok=True)
        input_path = os.path.join("application_data/input_image", "input_image.jpg")
        crop = self.frame[120:120+250, 200:200+250]
        cv2.imwrite(input_path, crop)
        self.log("Melakukan verifikasi...")

        results, matched_user = verify(self.model, return_identity=True)
        if matched_user:
            self.ui.verify_status.setStyleSheet("background-color: green; color: white")
            self.ui.verify_status.setText("OK")
            self.ui.nama.setText(f"WELCOME, {matched_user.upper()}")
            self.log(f"VERIFIED: {matched_user}")
        else:
            self.ui.verify_status.setStyleSheet("background-color: red; color: white")
            self.ui.verify_status.setText("NO")
            self.ui.nama.setText("WELCOME,")
            self.log("UNVERIFIED")

    def run_registration(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Register User", "Masukkan nama:")
        if not ok or not name.strip():
            self.log("Registrasi dibatalkan.")
            return
        name = name.strip().lower()
        anchor_path = os.path.join("data", name, "anchor")
        pos_path = os.path.join("data", name, "positive")
        os.makedirs(anchor_path, exist_ok=True)
        os.makedirs(pos_path, exist_ok=True)

        self.log(f"Mulai registrasi untuk {name}")
        self.ui.nama.setText(f"REGISTER: {name.upper()}")
        self.capture_sequence(anchor_path, "Anchor", 10)
        self.capture_sequence(pos_path, "Positive", 10)
        self.log(f"Registrasi selesai untuk {name}")
        self.ui.nama.setText(f"WELCOME, {name.upper()}")

    def capture_sequence(self, folder, label, duration):
        self.log(f"Ambil gambar {label.lower()} selama {duration} detik...")
        start_time = time.time()
        count = 0
        while time.time() - start_time < duration:
            QtWidgets.QApplication.processEvents()
            ret, frame = self.cap.read()
            if not ret:
                continue
            crop = frame[120:120+250, 200:200+250]
            filename = os.path.join(folder, f"{label.lower()}_{count}.jpg")
            cv2.imwrite(filename, crop)

            # Augmentasi otomatis
            for i, aug in enumerate(augment_image(crop)):
                aug_path = os.path.join(folder, f"{label.lower()}_{count}_aug{i}.jpg")
                cv2.imwrite(aug_path, aug)

            count += 1
            remaining = duration - int(time.time() - start_time)
            self.ui.countdown.setText(f"{remaining}s")
            time.sleep(0.3)
        self.ui.countdown.setText("")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FaceVerificationApp()
    window.show()
    sys.exit(app.exec_())
