import model
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
app = QApplication([])
app.setStyle('Fusion')

window = QWidget()
layout = QVBoxLayout()

startButton = QPushButton('Start')

def start():
    accuracy, choices = model.test_rule(model.search_rule(5000))
    accuracyLabel = QLabel("Accuracy: "+str(accuracy))
    choicesLabel = QLabel("Choices: "+str(choices))
    layout.addWidget(accuracyLabel)
    layout.addWidget(choicesLabel)

startButton.clicked.connect(start)

layout.addWidget(startButton)
window.setLayout(layout)
window.show()
app.exec_()

# TODO
# Show were the data is
# Start = Search rule (split rule und testing)
# Test = Accuracy + choices (better formatted -> maybe print sentences + evaluation) -> ask for feedback, make choice undoable
## better make it not a list but something like with a slider -> next, prev
# Ask = Ask a question from the found definitions in the test
# Make option in the corner to just run everything
# Extra option to test single sentence