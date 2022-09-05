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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MyWindow()
    window.show()

    #widget = MyWidget()
    #widget.show()

    sys.exit(app.exec_())