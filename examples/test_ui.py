import sys, time
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import * 

class MySignal(QObject):
    sig = Signal(str)

class MyLongThread(QThread):

    def __init__(self, parent = None):
        QThread.__init__(self, parent)
        self.exiting = False
        self.signal = MySignal()

    def run(self):
        end = time.time()+10
        while self.exiting==False:
                sys.stdout.write('*')
                sys.stdout.flush()
                time.sleep(1)
                now = time.time()
                if now>=end:
                        self.exiting=True
        self.signal.sig.emit('OK')

class MyThread(QThread):

    def __init__(self, parent = None):
        QThread.__init__(self, parent)
        self.exiting = False

    def run(self):
        while self.exiting==False:
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)

class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self,parent)
        self.centralwidget = QWidget(self)

        # Central layout 
        self.vbox = QVBoxLayout()

        # Input parameters area 
        self.parameters_area = QVBoxLayout()
        self.l1 = QLabel('Input Sequence')
        self.parameters_area.addWidget(self.l1)
        
        self.vbox.addLayout(self.parameters_area)

        self.output_area = QVBoxLayout()
        self.batchbutton = QPushButton('Load Sequence',self)
        self.longbutton = QPushButton('Run Analysis',self)
        self.label1 = QLabel('')
        self.label2 = QLabel('Add Input Values')
        

        self.lineedit = QLineEdit(parent=self)

        self.image = QImage('/projects/hidpy/output/trajectories.png')

        self.output_area.addWidget(self.batchbutton)
        
        self.output_area.addWidget(self.label1)
        self.output_area.addWidget(self.label2)

        

        self.l1 = QLabel('Parameter 1')
        self.output_area.addWidget(self.l1)
        self.output_area.addWidget(self.lineedit)

        self.output_area.addWidget(self.longbutton)
        
        
        self.lx = QLabel('Output Area')
        self.output_area.addWidget(self.lx)

        self.image_label = QLabel(" ")
        self.image_label.setPixmap(QPixmap.fromImage(self.image))

        self.output_area.addWidget(self.image_label)
        

        self.vbox.addLayout(self.output_area)



        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(self.vbox)
        self.thread = MyThread()
        
        
        '''
        self.longthread = MyLongThread()
        self.batchbutton.clicked.connect(self.handletoggle)
        self.longbutton.clicked.connect(self.longoperation)
        self.thread.started.connect(self.started)
        self.thread.finished.connect(self.finished)
        self.longthread.signal.sig.connect(self.longoperationcomplete)
        '''

        self.lineedit.textChanged.connect(self.printValue)

        
    def printValue(self):
        print(self.lineedit.text())

    def started(self):
        self.label1.setText('Continuous batch started')

    def finished(self):
        self.label1.setText('Continuous batch stopped')

    def terminated(self):
        self.label1.setText('Continuous batch terminated')

    def handletoggle(self):
        if self.thread.isRunning():
                self.thread.exiting=True
                self.batchbutton.setEnabled(False)
                while self.thread.isRunning():
                        time.sleep(0.01)
                        continue
                self.batchbutton.setText('Start batch')
                self.batchbutton.setEnabled(True)
        else:
                self.thread.exiting=False
                self.thread.start()
                self.batchbutton.setEnabled(False)
                while not self.thread.isRunning():
                        time.sleep(0.01)
                        continue
                self.batchbutton.setText('Stop batch')
                self.batchbutton.setEnabled(True)

    def longoperation(self):
        if not self.longthread.isRunning():
                self.longthread.exiting=False
                self.longthread.start()
                self.label2.setText('Long operation started')
                self.longbutton.setEnabled(False)

    def longoperationcomplete(self,data):
        self.label2.setText('Long operation completed with: '+data)
        self.longbutton.setEnabled(True)

if __name__=='__main__':
        app = QApplication(sys.argv)
        app.setApplicationDisplayName('Hidpy')

        window = MainWindow()
        window.show()
        sys.exit(app.exec_())