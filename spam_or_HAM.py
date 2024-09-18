import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
spam = pd.read_csv('/Users/zebra/Documents/projects/AI/spam.csv')
X = spam.iloc[0]
Y = spam.iloc[1]
data = train_test_split(X,Y,0.3)