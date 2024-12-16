import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer  


# CSV dosyasını okuma
df = pd.read_csv('Life Expectancy Data.csv')

missing_data = df[df.isnull().any(axis=1)]

print(missing_data)



missing_counts = df.isnull().sum()

print("Her sütundaki eksik veri sayıları:")
print(missing_counts)


from sklearn.impute import KNNImputer  

# "Life expectancy" kolonunu y olarak ayıralım (buradaki "Life expectancy" kolon adı veri setinize göre değişebilir)
y = df['Life expectancy ']  # Burada 'Life expectancy' hedef değişkenimiz
X = df.drop(columns=['Life expectancy ']).select_dtypes(include=['number'])  # Geri kalan tüm kolonları X'e aktaralım

impute_knn = KNNImputer(n_neighbors=3)
impute_knn.fit_transform(X)
print(X)

