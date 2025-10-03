# Canli yuz tespiti 1.1

'''
Yuz Tespiti1.0 'a yapilna guncellemeler:

1-)Goruntulerin sag alt kosesine logo eklemesi
2-)Koyu-Acik tema secimi ekleme
3-)kucuk performans iyilestirmeleri

tarih : 27.05.2025

'''


import sys
import cv2 as cv
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame,QPushButton
from PyQt6.QtGui import QImage, QPixmap ,QFont, QColor, QPalette
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import os

import os

haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')



class CustomWidget(QFrame):
    def __init__(self):
        super().__init__()

        self.content = QLabel(self)
        self.content.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo QLabel
        self.logo = QLabel(self.content)  # !!! Artık self değil, self.content üzerinde
        pixmap = QPixmap("/Users/yavuzselim/Desktop/opencv/photos/CBN-2.png").scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.logo.setPixmap(pixmap)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        self.logo.resize(w//15,h//20)
        self.logo.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.logo.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.logo.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.content.setGeometry(self.rect())  # content QLabel'i ana alanı kaplasın
   
        rect = self.rect()
        w = rect.width()
        h = rect.height()   

        margin_right = w//30
        margin_down = h//7

        x_logo = self.content.width() - self.logo.width() - margin_right
        y_logo = self.content.height() - self.logo.height() - margin_down
        self.logo.move(x_logo,y_logo)


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

        

        self.setWindowTitle("Yüz Tespit ve Sayımı")

        # Pencerenin arka plan rengini ayarla
        self.setStyleSheet("background-color: white;")  # veya 'white', '#f0f0f0', vb.


        self.display_width = 640
        self.display_height = 480

        self.image_label = CustomWidget()
        self.image_label.setFixedSize(self.display_width, self.display_height)

        self.image_label_original= CustomWidget()
        self.image_label_original.setFixedSize(self.display_width, self.display_height)

        self.face_count_label = QLabel(self)

        self.face_count_label.setFixedSize(self.display_width, 40)
        self.face_count_label.setText("Tespit edilen yüz sayısı: 0")
        self.face_count_label.setStyleSheet("color: black;")

        # Tema geçiş butonu
        self.theme_toggle_button = QPushButton("Koyu Tema")

        self.theme_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #ffd4aa;      /* Arka plan rengi */
                color: white;                   /* Yazı rengi */
                border: 2px solid black;     /* Kenarlık: kalınlık ve renk */
                border-radius: 10px;           /* Köşe yuvarlama, px cinsinden */
                padding: 8px 16px;             /* İç boşluk (yukarı-aşağı, sağ-sol) */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffaa56;     /* Üzerine gelince arka plan */
            }
            QPushButton:pressed {
                background-color: #1c5980;     /* Basılıyken arka plan */
            }
        """)
        self.theme_toggle_button.clicked.connect(self.toggle_theme)


        layout_V = QVBoxLayout()
        layout_H = QHBoxLayout()
        layout_H2 = QHBoxLayout()
        layout_H.addWidget(self.image_label)
        layout_H.addWidget(self.image_label_original)
        layout_V.addLayout(layout_H)
        layout_H2.addWidget(self.face_count_label)
        layout_H2.addStretch()
        layout_H2.addWidget(self.theme_toggle_button)
        layout_V.addLayout(layout_H2)
        self.setLayout(layout_V) 

                                # ic ice layout tanimlarken karisiklik cikabiliyormus farkli layout 
                                # gibi davranmiyor normal ayni layoutun icine widget eklemis gibi davraniyor

        # Thread Detected
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        # sinyal gelirse baglanacagi fonksiyonu soyluyor.
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
        self.image_label.content.setPixmap(scaled_pixmap)


    def update_image_original(self, cv_img):

        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.image_label_original.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label_original.content.setPixmap(scaled_pixmap)
   

    def update_face_count(self, count):
        self.face_count_label.setText(f"Tespit edilen yüz sayısı: {count}")

        font = QFont("Helvatica", 16)  
        font.setBold(True)        
        self.face_count_label.setFont(font) 

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.app.setPalette(dark_palette)

        self.face_count_label.setStyleSheet("color: white;")
        self.setStyleSheet("background-color: #2e2e2e;") 
        self.theme_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #ffd4aa;      /* Arka plan rengi */
                color: white;                   /* Yazı rengi */
                border: 2px solid white;     /* Kenarlık: kalınlık ve renk */
                border-radius: 10px;           /* Köşe yuvarlama, px cinsinden */
                padding: 8px 16px;             /* İç boşluk (yukarı-aşağı, sağ-sol) */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffaa56;     /* Üzerine gelince arka plan */
            }
            QPushButton:pressed {
                background-color: #1c5980;     /* Basılıyken arka plan */
            }
        """)

    def apply_light_theme(self):
        self.app.setPalette(self.app.style().standardPalette())

        self.face_count_label.setStyleSheet("color: black;")
        self.setStyleSheet("background-color: #f0f0f0;") 
        self.theme_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #ff7f00;      /* Arka plan rengi */
                color: white;                   /* Yazı rengi */
                border: 2px solid black;     /* Kenarlık: kalınlık ve renk */
                border-radius: 10px;           /* Köşe yuvarlama, px cinsinden */
                padding: 8px 16px;             /* İç boşluk (yukarı-aşağı, sağ-sol) */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffaa56;     /* Üzerine gelince arka plan */
            }
            QPushButton:pressed {
                background-color: #1c5980;     /* Basılıyken arka plan */
            }
        """)

    def toggle_theme(self):
        if self.theme_toggle_button.text() == "Koyu Tema":
            self.apply_dark_theme()
            self.theme_toggle_button.setText("Açık Tema")
        else:
            self.apply_light_theme()
            self.theme_toggle_button.setText("Koyu Tema")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.app = app  # Tema gecislerinde kullanilir.
    window.show()
    sys.exit(app.exec())

