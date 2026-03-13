# News Article Classification using Machine Learning

## Project Overview

This project implements a **Machine Learning pipeline** to automatically classify news articles into different categories such as:

- World
- Sports
- Business
- Sci/Tech

The goal of this project is to demonstrate how Natural Language Processing (NLP) techniques and machine learning models can be used to perform **text classification**.

The system processes raw news text, converts it into numerical features, trains multiple models, and selects the best-performing model based on accuracy.

---

## Dataset

The project uses the **AG News Dataset** obtained from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

The dataset contains thousands of labeled news articles belonging to the following categories:

| Label | Category |
|------|---------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

Each record contains:

- News Title
- News Description
- Category Label

---

## Machine Learning Pipeline

The project follows a structured machine learning workflow:

1. **Data Preprocessing**
2. **Text Feature Extraction**
3. **Model Training**
4. **Model Evaluation**
5. **Best Model Selection**

---

## Data Preprocessing

Before training the models, the text data is cleaned and processed.

The preprocessing steps include:

- Convert all text to lowercase
- Remove special characters and punctuation
- Remove stopwords (common words like "the", "is", "and")
- Tokenization of text

This step improves model performance by removing unnecessary noise from the data.

---

## Feature Engineering

Since machine learning models cannot directly process text data, the text is converted into numerical features using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

TF-IDF measures the importance of a word in a document relative to the entire dataset.

This results in a numerical representation of text that can be used by machine learning algorithms.

---

## Models Used

The following machine learning models are implemented and compared:

### 1. Logistic Regression
A widely used algorithm for classification problems that works well with high-dimensional data.

### 2. Naive Bayes
A probabilistic classifier commonly used for text classification tasks.

### 3. Linear Support Vector Machine (SVM)
A powerful classifier that finds the optimal hyperplane to separate classes.

---

## Model Evaluation

The models are evaluated using **accuracy on the test dataset**.

Performance comparison:

| Model | Accuracy |
|------|---------|
| Logistic Regression | ~90.8% |
| Naive Bayes | ~89% |
| Linear SVM | ~90.7% |

---

## Best Model

The best performing model is:

**Logistic Regression**

Final Accuracy:

**~90.8%**

---

## Project Folder Structure
