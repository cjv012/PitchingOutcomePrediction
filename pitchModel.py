import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasClassifier

df_pitches_description = pd.read_pickle('cleanedPitchData.pkl')
#print(df_pitches_description.info())

X_data = df_pitches_description[['pitch_type', 'release_speed', 'effective_speed',
       'spin_axis', 'spin_rate', 'release_extension', 'pitcher_handedness',
       'horizontal_movement (ft)', 'vertical_movement (ft)']]

Y_data = df_pitches_description['description']

X_binarized_pitch_data = pd.get_dummies(X_data, dtype=int)
Y_binarized_pitch_data = pd.get_dummies(Y_data, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X_binarized_pitch_data, Y_binarized_pitch_data, test_size=0.05, random_state=0)

std = StandardScaler()
train_X_standardized = pd.DataFrame(std.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
test_X_standardized = pd.DataFrame(std.transform(X_test), columns=X_test.columns, index=X_test.index)

