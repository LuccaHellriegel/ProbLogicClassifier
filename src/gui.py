import model
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox

app = QApplication([])
app.setStyle('Fusion')

window = QWidget()
window.setWindowTitle("ProbabilisticFlashcards")
layout = QVBoxLayout()

rule = None
choices = None

run_training_button = QPushButton('Run training')
training_label = QLabel("Path for training data:")
training_data_one = QLabel("Examples of definitions: ../data/definition_examples.txt")
training_data_two = QLabel("Examples of non-definitions: ../data/definition_negative_examples.txt")
training_accuracy = QLabel("Training accuracy: ")

def run_training():
    global rule
    train_accuracy, rule = model.search_rule(5000)
    training_accuracy.setText("Training accuracy: "+str(train_accuracy))

run_training_button.clicked.connect(run_training)

run_testing_button = QPushButton('Run testing')
testing_label = QLabel("Path for testing data:")
testing_data = QLabel("../data/definition_other.txt")
testing_accuracy = QLabel("Testing accuracy: ")
testing_choices = QLabel("Rule-decision: Is definition? ")

def run_testing():
    if rule is None:
        training_first = QMessageBox.question(window, 'No rule found!',
                                            "Run training first?",
                                            QMessageBox.Yes | QMessageBox.No)
        if training_first == QMessageBox.Yes:
            run_training()
            run_testing()
        else:
            pass
    else:     
        test_accuracy, choices = model.test_rule(rule)
        testing_accuracy.setText("Testing accuracy: "+str(test_accuracy))
        testing_choices.setText("Rule-decision: Is definition? "+str(choices))

run_testing_button.clicked.connect(run_testing)

run_all_button = QPushButton('Run all steps')

def run_all():
    run_training()
    run_testing()

run_all_button.clicked.connect(run_all)

widgets = [run_all_button, 
run_training_button, training_label, training_data_one, training_data_two, training_accuracy, 
run_testing_button, testing_label, testing_data, testing_accuracy, testing_choices]

for widget in widgets:
    layout.addWidget(widget)

window.setLayout(layout)
window.show()
app.exec_()
