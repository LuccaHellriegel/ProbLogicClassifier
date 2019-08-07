import model
import inspect
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox, QCheckBox, QLineEdit, QFormLayout, QGroupBox

model.refresh_data()

app = QApplication([])
app.setStyle('Fusion')

window = QWidget()
window.setWindowTitle("ProbLogicClassifier")
layout = QVBoxLayout()
formGroupBox = QGroupBox("Configuration: ")
formLayout = QFormLayout()
formGroupBox.setLayout(formLayout)
layout.addWidget(formGroupBox)

rule = None
choices = None

tau = QLineEdit("0.3")
formLayout.addRow(QLabel("Tau (influences probability of longer rules):"), tau)
complexity_lower_bound = QLineEdit("2")
formLayout.addRow(QLabel("Lower bound of rule complexity:"), complexity_lower_bound)
pyro_seed = QLineEdit("42")
formLayout.addRow(QLabel("Seed for pyro:"), pyro_seed)
training_steps = QLineEdit("5000")
formLayout.addRow(QLabel("Training steps:"), training_steps)

run_training_button = QPushButton('Run training')
training_label = QLabel("Path for training data:")
training_data_one = QLabel("Examples of definitions: ../data/definition_examples.txt")
training_data_two = QLabel("Examples of non-definitions: ../data/definition_negative_examples.txt")
training_accuracy = QLabel("Training accuracy: ")
training_rule = QLabel("Learned rule: ")
training_rule.setWordWrap(True)
training_complexity = QLabel("Final rule complexity: ")

def run_training():
    if use_same_seed_checkbox.isChecked():
        model.set_seed(int(pyro_seed.text()))
    model.tau = float(tau.text())
    model.complexity_lower_bound = float(complexity_lower_bound.text())
    if refresh_data_checkbox.isChecked():
        model.refresh_data()
    global rule
    train_accuracy, rule = model.search_rule(int(training_steps.text()))
    training_accuracy.setText("Training accuracy: "+str(train_accuracy))
    training_rule.setText("Learned rule: \n              "+model.rule_string)
    training_complexity.setText("Final rule complexity: "+str(model.rule_complexity))

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
        if refresh_data_checkbox.isChecked():
            model.refresh_data()     
        test_accuracy, choices = model.test_rule(rule)
        testing_accuracy.setText("Testing accuracy: "+str(test_accuracy))
        testing_choices.setText("Rule-decision: Is definition? "+str(choices))

run_testing_button.clicked.connect(run_testing)

refresh_data_checkbox = QCheckBox("Always refresh data?")
refresh_data_checkbox.setChecked(True)

use_same_seed_checkbox = QCheckBox("Always use same seed?")
use_same_seed_checkbox.setChecked(False)

refresh_data_button = QPushButton('Refresh data')

refresh_data_button.clicked.connect(model.refresh_data)

run_all_button = QPushButton('Run all steps')

def run_all():
    model.refresh_data()
    run_training()
    run_testing()

run_all_button.clicked.connect(run_all)

widgets = [refresh_data_checkbox, use_same_seed_checkbox, run_all_button, refresh_data_button, 
run_training_button, training_label, training_data_one, training_data_two, training_accuracy, training_rule, training_complexity, 
run_testing_button, testing_label, testing_data, testing_accuracy, testing_choices]

for widget in widgets:
    layout.addWidget(widget)

window.setLayout(layout)
window.show()
app.exec_()
