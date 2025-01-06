
# Importing  Libraries  

import numpy as np   
import pandas as pd   
import os  
import matplotlib.pyplot as plt  
import seaborn as sns  
from scipy import stats  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
 

df = pd.read_csv("c:\Users\fateme\Desktop\data.csv")  # Loading Data ---> this dataset is derived from Kaggle at (https://www.kaggle.com/datasets/warcoder/earthquake-dataset)
 

df['tsunami'] = df['tsunami'].replace({1: True, 0: False}) 
 
# Selecting specific columns for analysis  
selected_columns = ['title', 'alert', 'tsunami', 'country', 'magnitude', 'nst', 'mmi', 'sig', 'depth']  
data = df[selected_columns]  
 
numeric_cols = ['magnitude', 'nst', 'mmi', 'sig', 'depth']  # Identify numeric columns for outlier  

 
  
Q1 = data[numeric_cols].quantile(0.1)   
Q3 = data[numeric_cols].quantile(0.9)  
IQR = Q3 - Q1  
 

outlier_condition = ~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)  
filtered_data = data[outlier_condition]  # Removing outliers 

 
filtered_data.isnull().sum()  
cleaned_data = filtered_data.dropna()  
cleaned_data.isnull().sum()  
sampled_data = cleaned_data  # Keeping the cleaned data for further analysis  
 
# Data Visualizations  
fig_hist, (ax_magnitude, ax_nst, ax_mmi) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)  
 
 
hist_data = [('magnitude', ax_magnitude, "Magnitude"),  
             ('nst', ax_nst, "NST"),  
             ('mmi', ax_mmi, "MMI")]  
 

for data_col, axis, title in hist_data:  
    sns.histplot(data=sampled_data, x=data_col, ax=axis)  
    axis.set_title(title)  
 
plt.tight_layout()  
plt.show()  
 

fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=100)  
 
sns.histplot(data=sampled_data, x='sig', ax=axes[0])  
axes[0].set_title("Signal Strength (SIG)")  
 
sns.histplot(data=sampled_data, x='depth', ax=axes[1])  
axes[1].set_title("Depth")  
 
# Uncomment below lines for additional visualizations  
# sns.boxplot(data=sampled_data, x='tsunami', y='magnitude')  
# plt.xticks(rotation=45, ha="right")  
 
# sns.boxplot(data=sampled_data, x='alert', y='depth')  
# plt.xticks(rotation=45, ha="right")  
 

plt.figure(figsize=(15, 10))  
sns.heatmap(sampled_data[['magnitude', 'nst', 'mmi', 'sig', 'depth']].corr(),   
            annot=True, linecolor='black', cmap='magma')  
plt.show()  
 
# Preparing data for Linear Regression  
X = np.array(sampled_data.loc[:, 'sig'].values.reshape(-1, 1))   
Y = np.array(sampled_data.loc[:, 'magnitude'].values.reshape(-1, 1))   
 

plt.scatter(X, Y)  
plt.grid()  
plt.xlabel("Signal Strength (SIG)")  
plt.ylabel("Magnitude")  
 

linear_reg = LinearRegression()  # Creating a linear regression model  

 

linear_reg.fit(X, Y)  # Fitting the model to the data  
 
 
 
print('Slope (a) =', linear_reg.coef_[0][0])  
print('Intercept (b) =', linear_reg.intercept_[0])   
 
Y_predicted = linear_reg.predict(X)  

 
# errors  
error = Y_predicted - Y  
estimated_df = pd.DataFrame(np.concatenate((X, Y, Y_predicted, error), axis=1),   
                            columns=['Signal Strength (X)', 'Actual Magnitude (Y)', 'Predicted Magnitude (Y_hat)', 'Error (e)'])  
 
plt.scatter(X, Y, label='Actual Values', color='blue')  
plt.scatter(X, Y_predicted, color='red', label='Predicted Values')  
plt.grid()  
plt.legend(loc='upper right')  
 
plt.xlabel("Signal Strength (SIG)")  
plt.ylabel("Magnitude")  
 


plt.scatter(X, Y)  
plt.plot(X, Y_predicted, color='red')  
plt.grid()  
plt.xlabel("Signal Strength (SIG)")  
plt.ylabel("Magnitude")  
 
# Model Evaluation  
mse = mean_squared_error(Y, Y_predicted)  
r2 = r2_score(Y, Y_predicted)  
print("Mean Squared Error (MSE) =", mse, ", R-squared (R^2) =", r2)  
 
# Final result interpretation  
# It appears that the actual data points (blue) are widely spread and not closely following the red line.  
# This suggests that other factors might be influencing magnitude, or that the linear model may not effectively capture the relationship.  
 
 
# Additional Visualization 

# Pair Plot for Numeric Features  
sns.pairplot(sampled_data[numeric_cols])  
plt.suptitle("Pair Plot for Numeric Features", y=1.02)  
plt.show()