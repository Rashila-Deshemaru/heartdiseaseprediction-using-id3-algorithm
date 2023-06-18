# Heart Disease Prediction using ID3 Algorithm

This repository contains the implementation of a Heart Disease Prediction system using the ID3 (Iterative Dichotomiser 3) algorithm. The ID3 algorithm is a decision tree-based classification algorithm that utilizes entropy and information gain to build a tree model.

# Introduction

Heart disease is a critical health issue that affects a significant portion of the global population. Early detection and accurate prediction of heart disease can help in timely intervention and appropriate treatment planning. This project aims to develop a predictive model using the ID3 algorithm to identify the presence or absence of heart disease based on given input features.

# Dataset

The dataset used in this project is the Heart Disease UCI dataset from the UCI Machine Learning Repository. It consists of various medical attributes such as age, sex, cholesterol levels, blood pressure, etc., along with the target variable indicating the presence or absence of heart disease. The dataset is included in the repository as heart.csv.

# Installation

To run this project locally, please follow these steps:

Clone this repository:

``` git clone https://github.com/Rashila-Deshemaru/heartdiseaseprediction-using-id3-algorithm.git ```

Navigate to the project directory:

``` cd heartdiseaseprediction-using-id3-algorithm ```

Install the required dependencies. It is recommended to use a virtual environment:

``` pip install -r requirements.txt ```

# Usage

Once the installation is complete, you can run the heart_disease_prediction.py file to execute the heart disease prediction system. Make sure you have the dataset file heart.csv in the same directory.

``` python heart_disease_prediction.py ```

The program will train the ID3 algorithm on the provided dataset and prompt you to enter values for various features. Based on the input, it will predict whether the person is likely to have heart disease or not.

# Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

# License

This project is licensed under the MIT License. Feel free to use and modify the code as per your requirements.
