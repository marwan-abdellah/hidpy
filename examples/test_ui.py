import random
import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton,
                               QVBoxLayout, QWidget, QMainWindow)
from __feature__ import snake_case, true_property


class MyWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.hello = [
            "Hallo",
            "Hola",
            "Hei maailma",
            "Hola Mundo"
        ]

        self.button = QPushButton("Click me!")
        self.message = QLabel("Hello World")
        self.message.alignment = Qt.AlignCenter

        self.layout = QVBoxLayout(self)
        self.layout.add_widget(self.message)
        self.layout.add_widget(self.button)

        # Connecting the signal
        self.button.clicked.connect(self.magic)

    @Slot()
    def magic(self):
        self.message.text = random.choice(self.hello)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)

        # Set the central widget of the Window.
        self.setCentralWidget(button)

    def the_button_was_clicked(self):
        print("Clicked!")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    #widget = MyWidget()
    #widget.show()

    sys.exit(app.exec_())