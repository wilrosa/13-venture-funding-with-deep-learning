# 13-venture_funding_with_deep_learning

This repo contains the results of the module 13 challenge. To view the file, open the "venture_funding_with_deep_learning" folder and then the "venture_funding_with_deep_learning.ipynb" file. 

I assumed the role of an associate at a venture capital firm. The firm receives many pitch decks requesting funding from startups every day and asked me to create a model that predicts whether applicants will be successful if funded by the firm.

To begin, I used a CSV file containing more than 34,000 organizations that have received funding from the firm over the years. The CSV file contains a variety of information about each business, including whether or not it ultimately became successful. With my knowledge of machine learning and neural networks, I decided to use the features in the provided dataset to create a binary classifier model that will predict whether a startup requesting funding will become a successful business.

To predict whether the firm funding applicants will be successful, I created a binary classification model using a deep neural network and produced three technical deliverables:

(1) Preprocessed data for a neural network model.
(2) Used the model-fit-predict pattern to compile and evaluate a binary classification model.
(3) Optimized the model.

The accomplish these goals, I completed the following steps:

(1) Prepared the data for use on a neural network model.
(2) Compiled and evaluated a binary classification model using a neural network.
(3) Optimized the neural network model.

## Installation Guide

I needed to install the TensorFlow 2.0 library, which has several dependencies that should already be installed in the default Conda environment.

```python
!pip install --upgrade tensorflow
```
---

## Technologies

This project leverages python 3.7 with the following libraries and dependencies:

* [pandas](https://github.com/pandas-dev/pandas) - For manipulating data

* [numpy](https://github.com/numpy/numpy) - Fundamental package for scientific computing with Python

* [sklearn](https://github.com/scikit-learn/scikit-learn) - Module for machine learning built on top of SciPy

* [tensorflow](https://github.com/tensorflow/tensorflow) - End-to-end open source platform for machine learning

* [keras](https://github.com/keras-team/keras) - Deep learning API written in Python, running on top of the machine learning platform TensorFlow

---

### **Step 1: Prepared the Data for Use on a Neural Network Model**

Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed the dataset to use it to compile and evaluate the neural network model by completing the following data preparation steps:

(1) Read the `applicants_data.csv` file into a Pandas DataFrame. Reviewed the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.

(2) Dropped the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they were not relevant to the binary classification model.

(3) Encoded the dataset’s categorical variables using `OneHotEncoder`, and then placed the encoded variables into a new DataFrame.

(4) Added the original DataFrame’s numerical variables to the DataFrame containing the encoded variables. To complete this step, employed the Pandas `concat()` function that was introduced earlier in this course.

(5) Used the preprocessed data, created the features `(X)` and target `(y)` datasets. The targeted dataset was defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns defined the features dataset.

(6) Split the features and target sets into training and testing datasets.

(7) Used scikit-learn's `StandardScaler` to scale the features data.

### **Step 2: Compiled and Evaluated a Binary Classification Model Using a Neural Network**

Here, I used my knowledge of TensorFlow to design a binary classification deep neural network model. This model useed the dataset’s features to predict whether a startup funded by the firm will be successful based on the features in the dataset. I considered the number of inputs before determining the number of layers that the model should contain and the number of neurons on each layer. Then, I compiled and fit the model. Finally, I evaluated the binary classification model to calculate the model’s loss and accuracy.

To do so, I completed the following steps:

(1) Created a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras. Started with a two-layer deep neural network model that used the `relu` activation function for both layers.

(2) Compiled and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric. When fitting the model, started with a small number of epochs, in the 20-100 range.

(3) Evaluated the model using the test data to determine the model’s loss and accuracy.

(4) Saved and exported the model to an HDF5 file, and name the file `AlphabetSoup.h5`.
    
### **Step 3: Optimized the Neural Network Model**

In this step, I used my knowledge of TensorFlow and Keras to optimize the model to improve the its accuracy. I made two attempts to optimize the model.

To do so, I completed the following steps:

(1) Defined at least two new deep neural network models (resulting in the original model, plus two optimization attempts). With each, trying to improve on the first model’s predictive accuracy.

    Perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize the model for a predictive accuracy as close to 1 as possible, I experimented with the following techniques:

    (A) Adjusted the input data by dropping different features columns to ensure that no variables or outliers confused the model.
    (B) Added more neurons (nodes) to a hidden layer.
    (C) Added more hidden layers.
    (D) Used different activation functions for the hidden layers.
    (E) Added to or reduced the number of epochs in the training regimen.

(2) After finishing the models, I displayed the accuracy scores achieved by each model, and compared the results.

(3) Saved each of your models as an HDF5 file.

---
## Contributors

Brought to you by Wilson Rosa. https://www.linkedin.com/in/wilson-rosa-angeles/.

---
## License

MIT