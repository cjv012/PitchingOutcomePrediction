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
import keras_tuner as kt

df_pitches_description = pd.read_pickle('cleanedPitchData.pkl')
#print(df_pitches_description.info())

X_data = df_pitches_description[['pitch_type', 'release_speed', 'effective_speed',
       'spin_axis', 'spin_rate', 'release_extension', 'pitcher_handedness',
       'horizontal_movement (ft)', 'vertical_movement (ft)']]

Y_data = df_pitches_description['description']

X_binarized_pitch_data = pd.get_dummies(X_data, dtype=int)
Y_binarized_pitch_data = pd.get_dummies(Y_data, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X_binarized_pitch_data, Y_binarized_pitch_data, test_size=0.2, random_state=0)

std = StandardScaler()
train_X_standardized = pd.DataFrame(std.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
test_X_standardized = pd.DataFrame(std.transform(X_test), columns=X_test.columns, index=X_test.index)


#building out the model

def create_keras_model(num_hidden, activation='relu', optimizer='adam', loss='categorical_crossentropy', metrics='accuracy'):

    inputs = Input(shape=(train_X_standardized.shape[1]))
    dense = Dense(num_hidden, activation=activation)(inputs)
    hidden = Dense(num_hidden, activation=activation)(dense)
    outputs = Dense(y_test.shape[1],activation="softmax")(hidden)
    model = Model(inputs=inputs, outputs=outputs, name="model_2") # build the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics) # compile the model
    return model


#run a basic model

keras_model = KerasClassifier(model=create_keras_model, verbose=0, epochs=5, batch_size=128, model__num_hidden=20)
history = keras_model.fit(train_X_standardized, y_train, verbose=1,validation_data=(test_X_standardized, y_test))

#test_scores = keras_model.evaluate(test_X_standardized, y_test, verbose=2)
#print("Test loss:", test_scores[0])
#print("Test accuracy:", test_scores[1])
'''
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner = kt.Hyperband(model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)

tuner.search(train_X_standardized, y_train, batch_size=32, epochs=5, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps)'''

param_grid = {
    "model__optimizer" : ['adam', 'sgd'],
    "model__num_hidden": [100, 120],
    "epochs" : [175, 200],
    "batch_size" : [16, 32],
    "model__loss" : ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
} # define the parameter grid

# param_grid = {
#     "model__num_hidden": [100, 120],
#     "epochs" : [175, 200],
# } # define the parameter grid

#Hyperparameter Search

keras_model = KerasClassifier(model=create_keras_model, verbose=0, random_state=42) 
grid = GridSearchCV(keras_model, param_grid, return_train_score=True, cv=5, scoring='accuracy', n_jobs=-1, verbose=2) # define the GridSearchCV
grid_result = grid.fit(train_X_standardized, y_train) # fit the model using GridSearchCV

best_score = grid_result.best_score_ # Save the best score.
best_params = grid_result.best_params_ # Save the best parameters.
best_NN_classifier = grid_result.best_estimator_ # Save the best model.
print(f"Best Parameters: {best_params}")
print(f"Best Score: {round((best_score * 100), 2)}%")
history = best_NN_classifier.fit(train_X_standardized, y_train, verbose=1,validation_data=(test_X_standardized, y_test))