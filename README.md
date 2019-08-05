# ProbabilisticFlashcards

An experimental demo application for what a pipeline for automatically creating flashcards from a text using probabilistic programming could look like. 

## How To 

1. Download repo and open the folder in the command line

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Modify data 
* all data points are separated by a new line on Linux
* definition_examples.txt contains defintion training examples
* definition_negative_examples.txt contains training examples of non-definitions
* definition_other.txt contains testing data
* testing accuracy is calculated by assuming that the first half of the testing data is definitions and the second half is not

4. Go to the src folder and execute gui.py
```bash
cd src/
python3 gui.py
```

## Motivation

Flashcards enable us to study using two of the ways that research identified as working best overall: Active Recall and Spaced Repetition. 
However, there is usually a huge time-cost involved in translating study material to flashcards. Even though creating flashcards yourself gives an additional learning boost, in the presence of thousands of cards this often becomes negligible in my experience.

While software like Anki allows us to efficiently review our cards, there does not exist much software-based assistance for creating new cards.
That is the reason why I wanted to explore a lightweight pipeline for creating cards from text.
Restricting the type of cards to definitions is the obvious choice, as there are limited variants of sentences that contain a definition. 

## Current results

* As expected, classification of definitions is hugely dependant on training and test data
* Not meant as accurate classification but as a lightweight augmentation of the users learning process

## The Pipeline

Components:
* Model: Learn a rule to identify defintion-sentences
* Classifier: Use rule to classify definitions


### Learn what a definition is 

* Extract feature representation from sentences via sklearn's CountVectorizer
* Sample propositional logic expressions based on its ability to explain the training sentences
* Use rejection sampling to find expression that classifies the test data best

### Classify sentences

* Use logic expression to classify incoming sentences
* Present sentences to user for approval and identification of defined object

### Create and present probabilistic flashcards

* Save approved definition 
* Present definition to user by using one randomly chosen question format (e.g. How is X defined? What is the definition of X?)

## Background

TODO

### Why propositional logic?

TODO

### Why vectorization?

TODO

### Why flashcards?

TODO

### Why probabilistic programming?

TODO

* Using PP is much more lightweight than using full fledged neural nets
* Randomly and automatically generated educational material makes digital learning more effective
* Varying the questions breaks up the monotony of flashcards

### The goal of this application

TODO

* to sketch out an experiment with automatic flashcard generation
* to show some untapped potential in this area

