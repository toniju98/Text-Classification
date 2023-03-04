# Text-Classification

> This project shows a text classification. It works with news articles and predicts their categories. It's an supervised learning task.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)


## General Information
We work with text data. The process is the same like with other machine learning tasks. You first need to read in the data. The datafiles are .json files with news articles. After reading in the files to a dataframe you start to clean the data. Therefore we use spacy. For example there will be lowercased all articles and it will be reduced to nouns.
The reason is that to get the category of a text, the most meaning we get through nouns. After cleaning the data, the data will be vectorized to use it for machine learning. Therefore there is use the TFIDF-Vectorizer. 
After vectorizing the data the model will be trained. Therefore there can be tested many different models. After that the classification performance is evaluated.

## Technologies Used
- Python
- Spacy
- sklearn


## Features

- loading .json articles
- clean data
- predict news categories


## Screenshots




## Setup

You can run it with the command: python main.py



## Project Status
Project is: _complete_




## Room for Improvement

Room for improvement:
- new models
- other optimizations

To do:
- Update the html tags


## Acknowledgements
- This project was an college project in 2020



## Contact
Created by [@toniju98](https://github.com/toniju98) - feel free to contact me!
