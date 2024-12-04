# House Price Prediction Using Machine Learning

## Project Overview

This project aims to predict house prices in Ames, Iowa, based on various features using machine learning models. The dataset contains information about homes in Ames, including various attributes like square footage, neighborhood, quality of construction, and more. The goal is to predict the sale price of these homes using the data provided.

## Data Source

The dataset used for this project is from the [Kaggle House Price Prediction competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

- **train.csv**: The training data consisting of 1460 rows and 81 features, including the target variable `SalePrice`.
- **test.csv**: The test data consisting of 1459 rows, where the predictions for `SalePrice` need to be made.
- **sample_submission.csv**: A sample submission file showing the format for the final predictions.

## Project Steps

### 1. **Data Preprocessing**

   - **Missing Values Handling**: 
     - Numerical features with missing values were imputed with the median value.
     - Categorical features with missing values were imputed with the mode (most frequent value).
   - **One-Hot Encoding**: 
     - Categorical features were encoded using One-Hot Encoding to convert them into numerical format.
   - **Feature Scaling**: 
     - Numerical features were scaled using **StandardScaler** to standardize their values and bring them to a common scale.
   - **Dimensionality Reduction**: 
     - **PCA** (Principal Component Analysis) was used to reduce the dimensionality of the dataset while retaining the majority of variance.

### 2. **Model Development**

   - **Linear Regression**: Used as the baseline model to predict house prices.
   - **Random Forest Regressor**: An ensemble learning method used to improve model accuracy.
   - **XGBoost**: A gradient boosting algorithm used for better performance on large datasets.
   - **Hyperparameter Tuning**: Performed using **GridSearchCV** to find the best model parameters.

### 3. **Model Evaluation**

   - Models were evaluated using **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R²** score.
   - The best model was selected based on performance on the test set.

### 4. **Final Model**

   - The best performing model (e.g., XGBoost) was used to predict the target variable `SalePrice` on the test dataset.

## Files in this Repository

- **house_price_prediction.ipynb**: The Jupyter Notebook containing all the steps from data preprocessing to model training and evaluation.
- **train.csv**: The training data for model building.
- **test.csv**: The test data on which the model makes predictions.
- **sample_submission.csv**: Sample output submission file for the Kaggle competition format.
- **submission.csv**: The final output with predicted house prices for the test dataset.

## Instructions to Run the Code

1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/Nikitha130731/House-Price-Prediction-using-ML.git

2.	Install the required dependencies:

pip install -r requirements.txt


3.	Run the Jupyter notebook:
Open the Jupyter notebook house_price_prediction.ipynb in your local environment and run the cells sequentially.
4.	Model Training:
	•	The notebook will guide you through the preprocessing, model development, and evaluation steps.
	•	Once the best model is selected, it will generate predictions on the test dataset.
5.	Final Submission:
	•	A file submission.csv will be generated with the predicted sale prices.
	•	The file will have the format: Id and SalePrice.

Evaluation Metrics

	•	Mean Squared Error (MSE): Measures the average squared difference between the predicted and actual values.
	•	R² Score: The proportion of variance in the dependent variable (sale price) that is explained by the independent variables (features).

Future Work and Improvements

	•	Deep Learning Models: Exploring deep learning models like neural networks to improve prediction accuracy.
	•	Feature Engineering: Investigating additional feature creation or transformation methods to boost model performance.
	•	Advanced Algorithms: Experimenting with more advanced algorithms such as LightGBM or CatBoost for potential performance improvements.

License

This project is licensed under the MIT License - see the LICENSE file for details.


