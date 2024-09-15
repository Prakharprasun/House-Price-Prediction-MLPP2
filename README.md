# Ames Housing Price Prediction
This project implements an Ames Housing Price Prediction model using various machine learning techniques, including Neural Networks, Random Forests, and Gradient Boosting. The goal is to find the most accurate model.
Ames Housing dataset can be downloaded from : https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset

To run this project, you'll need the following libraries:

Python 3.x,
pandas,
numpy,
matplotlib,
scikit-learn,
tensorflow,
xgboost,

## Install the required packages using pip:

pip install pandas numpy matplotlib scikit-learn tensorflow xgboost

## Overview
The Ames Housing dataset is a comprehensive record of house sales in Ames, Iowa. We are trying to predict the sale price of houses using various regression techniques.
## Dataset
The dataset used is the Ames Housing dataset, it can be downloaded from : https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset
## Feature Engineering
Feature Engineering used in the project:

TotalArea: Sum of various area-related features such as Lot Area, Mas Vnr Area, Gr Liv Area, Garage Area, and Pool Area.

PriceperArea: SalePrice divided by TotalArea.
## Processing
One-Hot Encoding: Categorical variables are converted into numerical format using one-hot encoding.

Scaling: Numerical features are standardised using StandardScaler.

Imputation: Missing values are handled using SimpleImputer with the mean strategy.

One-hot encoding is an essential technique in data preprocessing for several reasons. It transforms categorical data into a format that machine learning models can easily understand and use. This transformation allows each category to be treated independently without implying any false relationships between them.
Additionally, many data processing and machine learning libraries support one-hot encoding. It fits smoothly into the data preprocessing workflow, making it easier to prepare datasets for various machine learning algorithms.

Label encoding is another method to convert categorical data into numerical values by assigning each category a unique number. However, this approach can create problems because it might suggest an order or ranking among categories that doesn't actually exist.
For example, assigning 1 to Red, 2 to Green, and 3 to Blue could make the model think that Green is greater than Red and Blue is greater than both. This misunderstanding can negatively affect the model's performance.

One-hot encoding solves this problem by creating a separate binary column for each category. This way, the model can see that each category is distinct and unrelated to the others. 
Label encoding is useful when the categorical data has an inherent ordinal relationship, meaning the categories have a meaningful order or ranking. In such cases, the numerical values assigned by label encoding can effectively represent this order, making it a suitable choice.


Scaling your data in machine learning (ML)is important because many algorithms use the Euclidean distance between two data points in their computations/derivations, which is sensitive to the scale of the variables. If one variable is on a much larger scale than another, that variable will dominate the distance calculation, and the algorithm will be affected by that variable more than the other. Scaling the data can help to balance the impact of all variables on the distance calculation and can help to improve the performance of the algorithm. In particular, several ML techniques, such as neural networks, require that the input data be normalised for it to work well.


There are several libraries in Python that can be used to scale data: 

Standardisation: The mean of each feature becomes 0 and the standard deviation becomes 1. 

Normalisation: The values of each feature are between 0 and 1. 

Min-Max Scaling: The minimum value of each feature becomes 0 and the maximum value becomes 1.
## Project Structure
AmesHousingModel Class: Encapsulates the entire workflow, from data loading to model evaluation.

load_and_prepare_data(): Loads the dataset and performs feature engineering.

split_data(): Splits the data into training, validation, and test sets.

plot_insights(): Visualises relationships between features and target variable.

train_neural_network(): Builds and trains a Neural Network model.

train_random_forest(): Builds and trains a Random Forest model.

train_gradient_boosting(): Builds and trains an XGBoost model.

evaluate_model(): Evaluates model performance using Mean Squared Error.

evaluate_performance(): Interprets the RMSE as a percentage of the average house price.

run(): Orchestrates the execution of all methods.

## Models
### Neural Network

Architecture:

Input layer matching the number of features.

Two hidden layers with 64 and 32 neurons respectively, using ReLU activation.

L2 regularisation to prevent overfitting.

Output layer with a single neuron for regression output.

Optimizer: Adam with a learning rate of 0.001.

Loss Function: Mean Squared Error (MSE).

Training: Runs for 100 epochs with a batch size of 100.

You have the regression equation y=Wx+b, where x is the input, W the weights matrix and b the bias.

Kernel Regularizer: Tries to reduce the weights 

Bias Regularizer: Tries to reduce the bias 

Activity Regularizer: Tries to reduce the layer's output thus will reduce the weights and adjust bias so Wx+b is smallest.

The L1 regularization penalty is computed as: loss = l1 * reduce_sum(abs(x))

The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))

Optimizers are algorithms or methods that are used to change or tune the attributes of a neural network such as layer weights, learning rate, etc. in order to reduce the loss and in turn improve the model.

Adam(Adaptive Moment Estimation) is an adaptive optimization algorithm that was created specifically for deep neural network training. It can be viewed as a fusion of momentum-based stochastic gradient descent and RMSprop. It scales the learning rate using squared gradients, similar to RMSprop, and leverages momentum by using the gradient’s moving average rather than the gradient itself, similar to SGD with momentum. Training and Validation The model is trained using the training set and evaluated on the validation set for 50 epochs using Mean Absolute Error (MAE) as a metric.

The model calculates the loss (or error) by comparing its prediction to the actual target value using a loss function. The loss function quantifies how far the model's prediction is from the target.

An epoch (also known as training cycle) in machine learning is a term used to describe one complete pass through the entire training dataset by the learning algorithm. During an epoch, the machine learning model is exposed to every example in the dataset once, allowing it to learn from the data and adjust its parameters (weights) accordingly. The number of epochs is a hyperparameter that determines the number of times the learning algorithm will work through the entire training dataset.
### Random Forest
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. Trees in the forest use the best split strategy, i.e. equivalent to passing splitter="best" to the underlying DecisionTreeRegressor.

Type: RandomForestRegressor from scikit-learn.

Parameters:
Number of estimators (trees): 100.

Features:
Handles non-linear relationships well.

Less prone to overfitting compared to decision trees.
### Gradient Boosting (XGBoost)
XGBoost is an optimised distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.

Gradient Boosting is a powerful boosting algorithm that combines several weak learners into strong learners, in which each new model is trained to minimise the loss function such as mean squared error or cross-entropy of the previous model using gradient descent. In each iteration, the algorithm computes the gradient of the loss function with respect to the predictions of the current ensemble and then trains a new weak model to minimise this gradient. The predictions of the new model are then added to the ensemble, and the process is repeated until a stopping criterion is met.

Type: XGBRegressor from the XGBoost library.

Parameters:
Objective: reg:squarederror for regression tasks.

Number of estimators: 1000.

Features:
Efficient handling of missing data.
Built-in regularization parameters to reduce overfitting.


## Results
After training and evaluating all models, the performance metrics are as follows:

Neural Network:

Train MSE: value

Test MSE: value

Random Forest:

Train MSE: value

Test MSE: value

Gradient Boosting (XGBoost):

Train MSE: value

Test MSE: value

The Gradient Boosting model performed the best, achieving the lowest Test MSE.
The RMSE as a percentage of the average house price is calculated to interpret the model's performance:
Based on the RMSE percentage, the model's performance is categorized as:

Perfect: RMSE ≤ 5%,
Great: 5% < RMSE ≤ 10%,
Good: 10% < RMSE ≤ 15%,
Fair: 15% < RMSE ≤ 20%,
Needs Improvement: RMSE > 20%,
## References
Data Camp

GeeksforGeeks

Edgeimpulse

Stackexchange

Keras

Sklearn

xgboost

