
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow, QLabel, QSizePolicy, QApplication, QAction, QHBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QEvent, QObject
from PyQt5.QtCore import *

import ctypes as C
import numpy as np
import torch
import os, sys, glob, cv2, hdf5storage, time

#import UIDesign

import models.dataset as ds
import helper as hp

import matplotlib as mpl
mpl.rc('image', cmap='inferno')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'
dtype = torch.cuda.FloatTensor
import time

# Import PyhtonNet
import clr

# Load IC Imaging Control .NET
#sys.path.append(os.getenv('IC35PATH') + "/redist/dotnet/x64")
clr.AddReference('TIS.Imaging.ICImagingControl35')
clr.AddReference('System')

# Import the IC Imaging Control namespace.
import TIS.Imaging
from System import TimeSpan
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


############################## Network loading ######################################

CapSize = 1536
Center = 768
ReconSize = 768

psf_file='./sample_data/psf5_2048.mat'
model_filepath='./trained_model/'
model, args = hp.load_model(psf_file, model_filepath, device = device)



########################################
class SubWindow(QWidget):
    
    def __init__(self):
        super(SubWindow, self).__init__()
        self.init_subui()
    
    def init_subui(self):

        self.ic = TIS.Imaging.ICImagingControl()
        self.snapsink = TIS.Imaging.FrameSnapSink(TIS.Imaging.MediaSubtypes.RGB32)
        self.ic.Sink = self.snapsink
        
        self.ic.LoadDeviceStateFromFile("device.xml",True)
        
    
        self.ax1 = QtWidgets.QLabel()
        self.ax2 = QtWidgets.QLabel()
       
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.ax1)
        self.layout.addWidget(self.ax2)
       
        self.setLayout(self.layout)
       
        self.show()
       
        self.ic.LiveStart()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_disp)
        self.timer.start()
        
    
    def update_disp(self):
        
        # start = time.time()
        self.frame = self.snapsink.SnapSingle(TimeSpan.FromSeconds(-1))
        
        imgcontent = C.cast(self.frame.GetIntPtr().ToInt64(),
                            C.POINTER(C.c_ubyte * self.frame.FrameType.BufferSize))
        
        img = np.ndarray(buffer=imgcontent.contents,
                         dtype=np.uint8,
                         shape=(self.frame.FrameType.Height,
                                self.frame.FrameType.Width,
                               int(self.frame.FrameType.BitsPerPixel / 8)))
        
        img_gray = cv2.cvtColor(img[:, 512:2560, :], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.flip(cv2.rotate(img_gray, cv2.ROTATE_180),1)
        raw = hp.crop_images(img_gray, args.psf_height_org, args.psf_width_org, args.psf_height_crop, args.psf_width_crop, args.psf_shift_height, args.psf_shift_width)
        
        img = torch.tensor(raw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        output = model((img).to(device))

        
        self.output_numpy = output[0].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
        output_numpy_crop = self.output_numpy[Center-ReconSize//2:Center+ReconSize//2, Center-ReconSize//2:Center+ReconSize//2]
        self.output_numpy = (output_numpy_crop-output_numpy_crop.min()) / output_numpy_crop.max()
        self.output_numpy[self.output_numpy <= 0] = 0
        
        self.output_numpy = 255 * self.output_numpy # Now scale by 255
        self.output_numpy = self.output_numpy.astype(np.uint8)
        self.capture = raw.astype(np.uint8)

        frame = cv2.cvtColor(self.capture, cv2.COLOR_BGR2RGB)
        recon = cv2.cvtColor(self.output_numpy, cv2.COLOR_BGR2RGB)
        
        height, width, bytesPerComponent = frame.shape
        bytesPerLine = bytesPerComponent * width
        q_image1 = QImage(frame.data,  width, height, bytesPerLine,
                               QImage.Format_RGB888).scaled(self.ax1.width(), self.ax1.height())
        self.ax1.setPixmap(QPixmap.fromImage(q_image1))
        
        height, width, bytesPerComponent = recon.shape
        bytesPerLine = bytesPerComponent * width

        q_image = QImage(recon.data,  width, height, bytesPerLine,
                               QImage.Format_RGB888).scaled(self.ax2.width(), self.ax2.height())
        self.ax2.setPixmap(QPixmap.fromImage(q_image))

       # self.ic.Dispose()
       # end = time.time()
       # print(end - start)

        
ic = TIS.Imaging.ICImagingControl()
snapsink = TIS.Imaging.FrameSnapSink(TIS.Imaging.MediaSubtypes.RGB32)
ic.Sink = snapsink

def SelectDevice():
    ic.LiveStop()
    ic.ShowDeviceSettingsDialog()
    if ic.DeviceValid is True:
       # ic.LiveStart()
        ic.SaveDeviceStateToFile("device.xml")
        
def ShowProperties():
    if ic.DeviceValid is True:
        ic.ShowPropertyDialog()
        ic.SaveDeviceStateToFile("device.xml")
        
def SnapImage():
    '''
    Snap and save an image
    '''
    image = snapsink.SnapSingle(TimeSpan.FromSeconds(1))
    TIS.Imaging.FrameExtensions.SaveAsBitmap(image,"test.bmp")
    
def Close():
    if ic.DeviceValid is True:
        ic.LiveStop()
    app.quit()
    

################################################

class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2000, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.Capture_txt = QtWidgets.QLabel(self.centralwidget)
        self.Capture_txt.setGeometry(QtCore.QRect(350, 20, 115, 41))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.Capture_txt.setFont(font)
        self.Capture_txt.setObjectName("Capture_txt")
        
        self.Capture_txt_2 = QtWidgets.QLabel(self.centralwidget)
        self.Capture_txt_2.setGeometry(QtCore.QRect(1050, 20, 355, 41))
        self.Capture_txt_2.setFont(font)
        self.Capture_txt_2.setObjectName("Capture_txt_2")
        
        self.ReconShow = QtWidgets.QWidget(self.centralwidget)
        self.ReconShow.setGeometry(QtCore.QRect(0, 50, 1650, 750))
        self.ReconShow.setObjectName("ReconShow")
        self.ReconShowac = SubWindow()
        layout = QtWidgets.QVBoxLayout(self.ReconShow)
        layout.addWidget(self.ReconShowac)
        
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(1660, 70, 81, 41))
        self.saveButton.setFont(font)
        self.saveButton.setObjectName("pushButton")
        
        snapAct =  QAction("Snap &Image",app)
        snapAct.triggered.connect(SnapImage)
        self.saveButton.addAction(snapAct)
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1205, 30))
        self.menubar.setObjectName("menubar")
        
        self.fileMenu = self.menubar.addMenu('&File')
        exitAct = QAction("&Exit",app)
        exitAct.setStatusTip("Exit program")
        exitAct.triggered.connect(Close)
        self.fileMenu.addAction(exitAct)
        
        self.deviceMenu = self.menubar.addMenu('&Device')
        devselAct = QAction("&Select",app)
        devselAct.triggered.connect(SelectDevice)
        self.deviceMenu.addAction(devselAct)

        devpropAct =  QAction("&Properties",app)
        devpropAct.triggered.connect(ShowProperties)
        self.deviceMenu.addAction(devpropAct)


        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Capture_txt.setText(_translate("MainWindow", "Capture"))
        self.Capture_txt_2.setText(_translate("MainWindow", "Real-time Reconstruction"))
        self.saveButton.setText(_translate("MainWindow", "Save"))
    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

