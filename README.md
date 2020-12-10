# GrapeMaster - a ML powered chatbot

This project was developed for the course of Cloud Applications and Systems at the University of Modena and Reggio Emilia during my MSc Degree.

## Project abstract

The goal of this project is the implementation of a recommendation system that suggests users with wine varieties based either on a generic description or a list of matching adjectives. The suggestion subsystem is based on the [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews) dataset, containing ~130k wine reviews and is powered by two separate Machine Learning models. The feature considered for suggestions is the `variety` feature. Since the target of the course is to focus on modern cloud architectures, the components of this project are structured for deployment on Google Cloud Platform.

## System architecture

The recommendation system has three main components:

 - Chatbot, to handle interaction with users. Implemented through GCP's Dialogflow service that streamlines the process of building a real-feeling conversation manager.
 - Classification models, to classify two kinds of requests coming from the user into the best fitting wine variety for their taste. Data cleaning and model training was done with `scikit-learn`, which is conveniently supported by GCP's AI Platform - Online Predictions service.
 - Fulfillment handler, to connect the chatbot with the ML models. A straightforward python script deployed on GCP's Cloud Functions.

![System architecture](https://github.com/enrico-ghidoni/grapemaster/blob/master/report/architecture.png)

## Classification models

The system relies on two different classification models to provide both a generic desciption based search and a more guided and first-timers friendly path.

The first model, `description-classification`, takes the `description` column as input feature. The goal of this model is to match a given description with the wine variety that most likely fits it. The problem is essentially a NLP task, the descriptions in the dataset are discretized with a *Term Frequency - Inverse Document Frequency* procedure after being cleaned of any unnecessary words.

For the second model, `adjective-classification`, a finite number of specific adjectives (collected from [here](https://winefolly.com/tips/wine-descriptions-chart-infographic/)) is extrapolated from the descriptions in order to obtain a *presence table* of every adjective in a description. The model takes a list of same adjectives collected by the chatbot to obtain a classification.

Due to the greatly imbalanced nature of the dataset against the `variety` feature, only varieties with a number of descriptions above the mean have been considered for training the models. To build the adjective-classification model a supersampling technique was applied to improve the final performance.

A step-by-step description for building the adjective-classification model is available in the `adjective-classification.ipynb` file. The same detailed description is not yet available for the `description-classification` model but is in the works.

## Repository content

 - `GrapeMaster.zip` is the exported configuration of the Dialogflow agent
 - `fulfillment` contains the python script that connects Dialogflow and the two models
 - `description-classification` contains the files related to data cleaning, model training and model deployment for the description-based model
 - `adjective-classification` contains the files related to data cleaning, model training and model deployment for the adjective-based model
 - `report` contains the material shown in this README as well as the project report that was delivered (at the moment only available in italian)