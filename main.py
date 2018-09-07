import cv2,os
import numpy as np
from PIL import Image
from multiprocessing.pool import ThreadPool
import threading
import sys
from PyQt5.QtWidgets import QTableWidget, QHeaderView, QTableWidgetItem, QHBoxLayout, QLabel, QFileDialog, QVBoxLayout, QListWidget, QListWidgetItem,QGridLayout, QPushButton, QWidget, QMessageBox, QMainWindow, QDesktopWidget, QAction, qApp, QMenu, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

global root

class TableRow():
    
    def __init__(self):
        super().__init__()
        hbox = QHBoxLayout()
        self.name = QLabel("")
        self.status = QLabel("waiting")
        hbox.addWidget(self.name)
        hbox.addStretch(1)
        hbox.addWidget(QLabel("|"))
        hbox.addStretch(1)
        hbox.addWidget(self.status)
        self.setLayout(hbox)
        
    def updateStatus(self, status):
        self.status.setText(status)

    def updateName(self, name):
        self.name.setText(name)
    
    

class FaceCrop(QWidget):
    IMAGES = []
    OUTPUT = None
    SORT = "alpha"
    
    def __init__(self, parent=None):
        super().__init__()
        self.setAcceptDrops(True)
        self.initUI()

        self.setStyleSheet("""
        QTableWidget {
            border: none;
            background-color: rgb(240, 240, 240);
        }
        QHeaderView::section {
        
        }
        """)

    def initUI(self):
        self.list_widget = QListWidget()
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(['name', 'attempts', 'crop', 'status'])
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.resizeSection(1, 70)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.resizeSection(2, 50)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.resizeSection(3, 70)
        
        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.clear)
        
        self.outputButton = QPushButton("Choose output folder")
        self.outputButton.clicked.connect(self.selectFile)
        
        self.startButton = QPushButton("Start")
        self.startButton.setEnabled(False)
        self.startButton.clicked.connect(self.preStart)

        self.output = QLabel()
        self.output.setText("No output folder selected")

        vbox = QVBoxLayout()
        vbox.addWidget(self.table_widget)
        vbox.addWidget(self.clearButton)
        vbox.addWidget(self.outputButton)
        vbox.addWidget(self.startButton)
        vbox.addWidget(self.output)
        
        self.setLayout(vbox)
        
        self.setGeometry(300, 300, 400, 600)    
        self.center()
            
        self.setWindowTitle('Face Crop')    
        self.show()

    def selectFile(self):
        self.OUTPUT = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.output.setText("Saving to: {}".format(self.OUTPUT))
        if len(self.IMAGES) > 0:
            self.startButton.setEnabled(True)

    def preStart(self):
        threading.Thread(target=self.start).start()
        
    def start(self):
        for image in self.IMAGES:
            image['output'] = self.OUTPUT
        pool = ThreadPool(4)
        results = pool.map(find_faces, self.IMAGES)
        pool.close()
        pool.join()
        self.IMAGES = results
        self.output.setText("Complete! Processed {} images".format(len(self.IMAGES)))
        self.SORT = "cropped"
        self.addAllImages(self.IMAGES)
        
        
    def clear(self):
        self.IMAGES = []
##        self.list_widget.clear()
        self.table_widget.setRowCount(0)
        self.startButton.setEnabled(False)
        self.OUTPUT = None
        self.output.setText("No output folder selected")
        self.SORT = "alpha"

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            if self.OUTPUT:
                self.startButton.setEnabled(True)
            for url in event.mimeData().urls():
                path = str(url.toLocalFile())
                name = ((path.split("/")[-1]).split(".")[0]).replace("_"," ")
                if path not in [image['path'] for image in self.IMAGES]:
                    self.IMAGES.append({"path": path, "name": name, "status": "waiting", "attempts": 0, "cropped": 100})
            self.addAllImages(self.IMAGES)
            event.accept()
        else:
            event.ignore()

    def addAllImages(self, images):
        self.table_widget.setRowCount(0)
        if self.SORT == "cropped":
            images = sorted(images, key=lambda k: k['cropped'])
        else:
            images = sorted(images, key=lambda k: k['name'])
        self.table_widget.setRowCount(len(images))
        for i, image in enumerate(images):
            image['index'] = i
            self.updateTableRow(image)
            
    def updateTableRow(self, image):
        self.table_widget.removeRow(image['index'])
        self.table_widget.insertRow(image['index'])
        self.table_widget.setItem(image['index'],0,QTableWidgetItem(image["name"]))
        cell1 = QTableWidgetItem(str(image["attempts"]))
        cell1.setTextAlignment(Qt.AlignCenter)
        self.table_widget.setItem(image['index'],1,cell1)
        cell2 = QTableWidgetItem("{}%".format(image["cropped"]))
        cell2.setTextAlignment(Qt.AlignCenter)
        self.table_widget.setItem(image['index'],2,cell2)
        cell3 = QTableWidgetItem(image["status"])
        cell3.setTextAlignment(Qt.AlignCenter)
        self.table_widget.setItem(image['index'],3,cell3)
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
            
    def contextMenuEvent(self, event):
   
       cmenu = QMenu(self)
       
       newAct = cmenu.addAction("New")
       opnAct = cmenu.addAction("Open")
       quitAct = cmenu.addAction("Quit")
       action = cmenu.exec_(self.mapToGlobal(event.pos()))
       
       if action == quitAct:
           qApp.quit()

def cropFace(face, original, padding):
    if face['y'] - padding < 0 or face['y'] + face['h'] + padding > original.shape[0]: # Y crop out of bounds 
        return cropFace(face, original, padding - 1)
    elif face['x'] - padding < 0 or face['x'] + face['w'] + padding > original.shape[1]: # X crop out of bounds
        return cropFace(face, original, padding - 1)
    else:
        # Found a padding that is within the image borders. Cropping original image.
        return original[
            face['y']-padding:face['y']+face['h']+padding,
            face['x']-padding:face['x']+face['w']+padding]
    #end if
#end cropFace

if hasattr(sys, "_MEIPASS"):
    classifier = os.path.join(sys._MEIPASS, "haarcascade\haarcascade_frontalface_default.xml")
else:
    classifier = "haarcascade_frontalface_default.xml"

def find_faces(image):
    if image['status'] != 'waiting':
        return image
    fc.updateTableRow(image)
    detector = cv2.CascadeClassifier(classifier)
    imagePath = image['path']
    minSize = (50,50)
    scaleFactor = 1.01
    minNeighbors = 30
    pilImage = Image.open(imagePath).convert('RGB')
    colorImg = np.array(pilImage,'uint8')
    grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    name = os.path.split(imagePath)[-1].split(".")[0]
    foundMultipleFaces = True
    while foundMultipleFaces:
        image['attempts'] += 1
        fc.updateTableRow(image)
        faces = detector.detectMultiScale(grayImg,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize)
        if len(faces) == 1:
            foundMultipleFaces = False
        else:
            if minNeighbors > 1:
                if scaleFactor < 1.2:
                    scaleFactor += 0.01
                else:
                    minNeighbors -= 1
                    scaleFactor = 1.01
            else:
                foundMultipleFaces = False
                
            
    for i, (x,y,w,h) in enumerate(faces):
        face = {"x": x, "y": y, "w": w, "h": h}
        imageName = name
        if len(faces) > 1:
            imageName = "{}-{}".format(name, i)
        #end if
        pad = int(face['w'] / 2)
        imageArray = cropFace(face, colorImg, pad)
        img = Image.fromarray(imageArray, 'RGB')
        image['face'] = face
        image['originalSize'] = pilImage.size
        image['cropSize'] = img.size
        image['img'] = img
        image['cropped'] = round( (((img.size[0] * img.size[1]) / ((face['h'] * 2) * (face['w'] * 2)))) * 1000) / 10
        resize_and_save(image['output'], img, imageName)
        image['status'] = 'complete'
        fc.updateTableRow(image)
        return image
        return (face,pilImage.size,img.size)
        #self.resize_and_save(img, imageName)
    #end for
#end find_faces
            
def resize_and_save(output, image, imageName):
    image = image.resize((160,160), Image.ANTIALIAS)
    image.save("{}/{}.jpg".format(output,imageName),optimize=True)

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    fc = FaceCrop()
    sys.exit(app.exec_())
