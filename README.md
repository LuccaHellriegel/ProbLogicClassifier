# ProbabilisticFlashcards

A demo application for what a pipeline for automatically creating flashcards from a text using probabilistic programming could look like. 

TODO: Table of Contents

## How To 

TODO

## The Pipeline

Components:
* Learner: Learn a rule to identify defintion-sentences
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

* Randomly and automatically generated educational material makes digital learning more effective
* Varying the questions breaks up the monotony of flashcards

### The goal of this application

TODO

### Possible improvements

* Online learning: Use user-feedback (acceptance of found definition-sentence) to generate new classification rule
* Automatically identify defined object in the sentence
