# ProbabilisticFlashcards

An experimental demo application for what a pipeline for automatically creating flashcards from a text using probabilistic programming could look like. 

TODO: Table of Contents

## Summary

## How To 

TODO

1. Download repo and open the folder in the command line

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Go to the src folder and execute gui.py
```bash
cd src/
python3 gui.py
```

## Current results

* Hugely dependant on training and test data
* Not meant as accurate classification but as a lightweight augmentation of the users learning process

## The Pipeline

Components:
* Model: Learn a rule to identify defintion-sentences
* Classifier: Use rule to classify definitions
* Generator: Create definition-flashcards based an a predefined schema and present it to the user

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

### Possible improvements

* Online learning: Use user-feedback (acceptance of found definition-sentence) to generate new classification rule
* Automatically identify defined object in the sentence
