import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


df = pd.read_csv('Life Expectancy Data.csv')

missing_data = df[df.isnull().any(axis=1)]

print(missing_data)



missing_counts = df.isnull().sum()

print("Her sütundaki eksik veri sayıları:")
print(missing_counts)

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))  # Only scale numeric data
scaled_df = pd.DataFrame(scaled_data, columns=df.select_dtypes(include=[np.number]).columns)

for col in scaled_df.columns:
    if scaled_df[col].isnull().sum() > 0:  # If the column has missing values
        print(f"Imputing missing values for column: {col}")
        
        # Separate rows with and without missing values
        train_data = scaled_df[scaled_df[col].notnull()]
        test_data = scaled_df[scaled_df[col].isnull()]
        
        # Use other columns as features
        X_train = train_data.drop(columns=[col]).values
        y_train = train_data[col].values
        X_test = test_data.drop(columns=[col]).values
        
        # Handle missing values in X_train and X_test
        valid_train_rows = ~np.isnan(X_train).any(axis=1)  # Remove rows with NaN in X_train
        X_train = X_train[valid_train_rows]
        y_train = y_train[valid_train_rows]
        
        # Temporarily fill NaNs in X_test with column means (or another strategy)
        X_test = np.nan_to_num(X_test, nan=np.nanmean(X_train, axis=0))
        
        # Define the Gaussian Process model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
        
        # Fit the model
        gpr.fit(X_train, y_train)
        
        # Predict the missing values
        y_pred, y_std = gpr.predict(X_test, return_std=True)
        
        # Fill the missing values
        scaled_df.loc[scaled_df[col].isnull(), col] = y_pred
imputed_data = scaler.inverse_transform(scaled_df)
imputed_df = pd.DataFrame(imputed_data, columns=df.select_dtypes(include=[np.number]).columns)

# Replace the original numeric columns with imputed values
for col in imputed_df.columns:
    df[col] = imputed_df[col]



missing_counts = df.isnull().sum()

print("Her sütundaki eksik veri sayıları:")
print(missing_counts)