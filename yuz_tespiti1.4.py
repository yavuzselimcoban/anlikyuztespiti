# Canli yuz tespiti 1.2

'''
Yuz Tespiti1.3 'e yapilna guncellemeler:

1-)dnn modelin yüklenmesi her karede yapılıyor iken init içerisine alınıp performans iyileştirilmesi yapıldı.
   (fps de kayda değer artış)

tarih : 15.09.2025

'''

import cv2 as cv
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame,QPushButton
from PyQt6.QtGui import QImage, QPixmap ,QFont, QColor, QPalette
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import os
import time
import sys

class Config:
    # DNN model yolları (proje kök dizininde olduğunu varsayarak)
    DNN_PROTOTXT = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
    DNN_MODEL = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")

    @classmethod
    def load_dnn_model(cls):
        if not os.path.exists(cls.DNN_PROTOTXT):
            raise FileNotFoundError(f"{cls.DNN_PROTOTXT} bulunamadi !!!")
        if not os.path.exists(cls.DNN_MODEL):
            raise FileNotFoundError(f"{cls.DNN_MODEL} bulunamadi !!!")
        return cv.dnn.readNetFromCaffe(cls.DNN_PROTOTXT, cls.DNN_MODEL)


class CustomWidget(QFrame):
    def __init__(self):
        super().__init__()

        self.content = QLabel(self)
        self.content.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo QLabel
        self.logo = QLabel(self.content)  # !!! Artık self değil, self.content üzerinde
        pixmap = QPixmap("photos/CBN-2.png").scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
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

    def __init__(self):
        super().__init__()
        self._run_flag = True
        

    def run(self):
        cap = cv.VideoCapture(0)
        prev_time = 0
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            width = int(frame.shape[1])
            height = int(frame.shape[0])
            dim = (width, height)
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            frame = cv.flip(frame, 1)

            # fps hesabi
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            cv.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (191, 0, 191), 2)

            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(rgb_image)


        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)
    face_count_signal = pyqtSignal(int)  # Yüz sayısı sinyali

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.net = Config.load_dnn_model()
    def run(self):

        cap = cv.VideoCapture(0)
        prev_time = 0
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # Kare boyutları
            h, w = frame.shape[:2]
            frame = cv.flip(frame, 1)
            # Blob oluştur
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            # DNN (Deep Neural Network) girişine uygun formatta bir blob (görüntü verisi) oluşturur.
            # cv.resize(frame, (300, 300)) → modelin beklediği boyuta indirger.
            # (104.0, 177.0, 123.0) → ortalama değerler; görüntüden çıkarılır (ön işleme).
            
            self.net.setInput(blob)     
            # hazirladigimiz blob u modele verdik.     
            detections = self.net.forward()
            # bu degisken tespitn edilen yuzler icerir

            # Yüzleri işle
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2] # bu bize confidence i donuyor
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    if confidence > 0.9:
                        color = (0, 255, 0) # yesil
                    elif confidence > 0.7:
                        color = (0, 255, 255)  # sari
                    else:
                        color = (0, 0, 255) # kirmizi

                    # Yüz kutusu çiz
                    cv.rectangle(frame, (startX, startY), (endX, endY),
                                color,2)

                    # Güven skoru yaz
                    text = f"{confidence*100:.1f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv.putText(frame, text, (startX, y),
                            cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # fps hesabi
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            cv.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (191, 0, 191), 2)
            
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


            face_count = 0  # <--- Her kare için yüz sayacı

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    face_count += 1


            self.change_pixmap_signal.emit(rgb_image)
            self.face_count_signal.emit(face_count)

        # Kaynakları serbest bırak
        cap.release()


    def stop(self):
        self._run_flag = False
        self.wait()

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
                background-color: #6B8E23;      /* Arka plan rengi */
                color: #F0F0F0;                   /* Yazı rengi */
                border: none;     /* Kenarlık: kalınlık ve renk */
                border-radius: 10px;           /* Köşe yuvarlama, px cinsinden */
                padding: 8px 16px;             /* İç boşluk (yukarı-aşağı, sağ-sol) */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #556B2F;     /* Üzerine gelince arka plan */
            }
            QPushButton:pressed {
                background-color: #4A581B;     /* Basılıyken arka plan */
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

        font = QFont("Helvetica", 16)  
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
                background-color: #556270;      /* Arka plan rengi */
                color: #E0E0E0;                   /* Yazı rengi */
                border: 2px solid #708090;     /* Kenarlık: kalınlık ve renk */
                border-radius: 10px;           /* Köşe yuvarlama, px cinsinden */
                padding: 8px 16px;             /* İç boşluk (yukarı-aşağı, sağ-sol) */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3B4A58;     /* Üzerine gelince arka plan */
            }
            QPushButton:pressed {
                background-color: #2C3E50;     /* Basılıyken arka plan */
            }
        """)

    def apply_light_theme(self):
        self.app.setPalette(self.app.style().standardPalette())

        self.face_count_label.setStyleSheet("color: black;")
        self.setStyleSheet("background-color: #f0f0f0;") 
        self.theme_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #6B8E23;      /* Arka plan rengi */
                color: #F0F0F0;                   /* Yazı rengi */
                border: none;     /* Kenarlık: kalınlık ve renk */
                border-radius: 10px;           /* Köşe yuvarlama, px cinsinden */
                padding: 8px 16px;             /* İç boşluk (yukarı-aşağı, sağ-sol) */
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #556B2F;     /* Üzerine gelince arka plan */
            }
            QPushButton:pressed {
                background-color: #4A581B;     /* Basılıyken arka plan */
            }
        """)

    def toggle_theme(self):
        if self.theme_toggle_button.text() == "Koyu Tema":
            self.apply_dark_theme()
            self.theme_toggle_button.setText("Açık Tema")
        else:
            self.apply_light_theme()
            self.theme_toggle_button.setText("Koyu Tema")

    def closeEvent(self, event):
        self.thread.stop()
        self.thread_ori.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.app = app  # Tema gecislerinde kullanilir.
    window.show()
    sys.exit(app.exec())