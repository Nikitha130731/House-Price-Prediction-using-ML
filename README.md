
# House Price Prediction using Machine Learning

## Project Overview

The House Price Prediction project aims to develop a machine learning model that accurately predicts house prices based on various features from a dataset. This project utilizes Python and popular libraries such as Pandas, Scikit-learn, and Matplotlib for data analysis and visualization.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Getting Started](#getting-started)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [License](#license)

## Installation

To run this project, you need to have Python 3.x installed. You can install the necessary libraries using the following command:


**--pip install pandas numpy scikit-learn matplotlib seaborn**


## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/input). It contains detailed information about houses in Ames, Iowa, with 1,460 rows and 81 columns, featuring various attributes like house area, quality, and neighborhood.

## Features

The main features used for predicting house prices include:

- **OverallQual**: Overall quality of the house
- **GrLivArea**: Above-ground living area in square feet
- **GarageCars**: Size of garage in car capacity
- **YearBuilt**: Year the house was built

Additional features from the dataset may be explored and utilized as necessary.

## Getting Started

1. Clone the repository to your local machine:

   
    git clone https://github.com/Nikitha130731/House-Price-Prediction-using-ML.git
  

2. Navigate to the project directory:

  
    cd House-Price-Prediction-using-ML
  

3. Open the Jupyter Notebook:

    jupyter notebook
  

4. Run the Jupyter Notebook `**House Price Prediction.ipynb**` to explore the project.

## Modeling

In this project, we have implemented a Linear Regression model to predict house prices. The model is trained on a portion of the dataset and evaluated on a test set to assess its performance.


from sklearn.linear_model import LinearRegression

# Create and train the model
lr_model = LinearRegression()
lr_model.fit(X_train_encoded, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_encoded)


## Evaluation

The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and R-squared:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


## Visualization

Visualizations are included to analyze the dataset and the model's predictions. Various plots are generated using Matplotlib and Seaborn to provide insights into the data.

import matplotlib.pyplot as plt
import seaborn as sns

# Example visualization
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()



