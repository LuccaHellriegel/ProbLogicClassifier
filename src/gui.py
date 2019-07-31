import model
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
app = QApplication([])
app.setStyle('Fusion')

window = QWidget()
layout = QVBoxLayout()

startButton = QPushButton('Start')

def start():
    accuracy, choices = model.search_rule(5000)
    accuracyLabel = QLabel("Accuracy: "+str(accuracy))
    choicesLabel = QLabel("Choices: "+str(choices))
    layout.addWidget(accuracyLabel)
    layout.addWidget(choicesLabel)

startButton.clicked.connect(start)

layout.addWidget(startButton)
window.setLayout(layout)
window.show()
app.exec_()