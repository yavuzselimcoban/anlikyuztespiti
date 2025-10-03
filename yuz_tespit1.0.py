# Canli yuz tespiti ve sayimi 1.0 

''''
tarih : 25.05.2025
'''

import sys
import cv2 as cv
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout,QHBoxLayout,QFrame
from PyQt6.QtGui import QImage, QPixmap ,QFont
from PyQt6.QtCore import QThread, pyqtSignal, Qt


haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


class VideoThread_ori(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            width = int(frame.shape[1])
            height = int(frame.shape[0])
            dim = (width, height)
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            frame = cv.flip(frame, 1)


            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(rgb_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()


class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)
    face_count_signal = pyqtSignal(int)  # Yüz sayısı sinyali

    def run(self):

        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            width = int(frame.shape[1] * 0.3)
            height = int(frame.shape[0] * 0.3)
            dim = (width, height)
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            frame = cv.flip(frame, 1)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

            # Yüzleri çerçeve içine al
            for (x, y, w, h) in faces_rect:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Yüz sayısını gönder
            self.face_count_signal.emit(len(faces_rect))

            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(rgb_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("Yüz Tespit ve Sayimi")

        self.display_width = 640
        self.display_height = 480

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.display_width, self.display_height)

        self.image_label_original= QLabel(self)
        self.image_label_original.setFixedSize(self.display_width, self.display_height)

        self.face_count_label = QLabel(self)
        self.face_count_label.setFixedSize(self.display_width, 40)
        self.face_count_label.setText("Tespit edilen yüz sayısı: 0")

        layout_V = QVBoxLayout()
        layout_H = QHBoxLayout()
        layout_H.addWidget(self.image_label)
        layout_H.addWidget(self.image_label_original)
        layout_V.addLayout(layout_H)
        layout_V.addWidget(self.face_count_label)
        self.setLayout(layout_V)

        # Thread Detected
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.face_count_signal.connect(self.update_face_count)
        self.thread.start()

        # Thread Original
        self.thread_ori = VideoThread_ori()
        self.thread_ori.change_pixmap_signal.connect(self.update_image_original)
        self.thread_ori.start()


    def update_image(self, cv_img):

        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)


    def update_image_original(self, cv_img):

        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.image_label_original.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label_original.setPixmap(scaled_pixmap)     


    def update_face_count(self, count):
        self.face_count_label.setText(f"Tespit edilen yüz sayısı: {count}")

        font = QFont("Helvatica", 16)  
        font.setBold(True)        
        self.face_count_label.setFont(font) 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
