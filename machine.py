import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from pandas.api.types import is_numeric_dtype
# import warnings
# from sklearn import tree
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.tree  import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC, LinearSVC
# from sklearn.naive_bayes import BernoulliNB
# from lightgbm import LGBMClassifier
# from sklearn.feature_selection import RFE
# import itertools
# from xgboost import XGBClassifier
# from tabulate import tabulate

train=pd.read_csv('Train_data.csv')

test=pd.read_csv('Test_data.csv')



total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")

print(f"Number of duplicate rows: {train.duplicated().sum()}")

sns.countplot(x=train['Label'])

print('Class distribution Training set:')
print(train['Label'].value_counts())

def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

le(train)
le(test)

# train.drop(['num_outbound_cmds'], axis=1, inplace=True)
# test.drop(['num_outbound_cmds'], axis=1, inplace=True)

p=train['Label'].value_counts()



train['service'].value_counts()

X_train = train.drop(['Label'], axis=1)
Y_train = train['Label']





from sklearn.decomposition import PCA

# Select the features you want to apply PCA on
useful_features = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Subset the DataFrame with only the selected features
X_selected = X_train[useful_features]

# Initialize PCA with the desired number of components
pca = PCA(n_components=10)

# Fit PCA to the selected features
X_selected_pca = pca.fit_transform(X_selected)

# Get the names of the original features that contribute the most to each principal component
selected_features = []
for i in range(X_selected_pca.shape[1]):
    component_index = useful_features[np.abs(pca.components_[i]).argsort()[-1]]
    selected_features.append(component_index)

# Print the selected features
print("Selected features:", selected_features)

X_train = X_train[selected_features]

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.80, random_state=2)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score

# Get the number of features from the shape of the training data
num_features = x_train.shape[1]

# Define the neural network model
model = Sequential()

# Add layers with increasing number of neurons
neurons = [16, 32, 64, 128]
for num_neurons in neurons:
    if num_neurons == neurons[0]:
        # First layer with input shape
        model.add(Dense(num_neurons, activation='relu', input_shape=(num_features,)))
    else:
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(0.5))

# Add layers with decreasing number of neurons
neurons.reverse()
for num_neurons in neurons:
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', test_accuracy)

# Make predictions
# y_pred_prob = model.predict(x_test)
# print(x_test)
# y_pred = (y_pred_prob > 0.5).astype(int)




import matplotlib.pyplot as plt

# Train the model and store the training history
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot training/validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training/validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.metrics import accuracy_score

# # Get the number of features from the shape of the training data
# num_features = x_train.shape[1]

# # Define the neural network model
# model = Sequential()

# # Add layers with increasing number of neurons
# for num_neurons in range(8, 513, 64):
#     if num_neurons == 8:
#         # First layer with input shape
#         model.add(Dense(num_neurons, activation='relu', input_shape=(num_features,)))
#     else:
#         model.add(Dense(num_neurons, activation='relu'))
#     model.add(Dropout(0.5))

# # Add layers with decreasing number of neurons
# for num_neurons in range(448, 7, -64):
#     model.add(Dense(num_neurons, activation='relu'))
#     model.add(Dropout(0.5))

# # Output layer
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# # Train the model
# history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model on test data
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print('Test accuracy:', test_accuracy)

# # Make predictions
# y_pred_prob = model.predict(x_test)
# y_pred = (y_pred_prob > 0.5).astype(int)

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.metrics import accuracy_score

# # Get the number of features from the shape of the training data
# num_features = x_train.shape[1]

# # Define the neural network model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(num_features,)),
#     Dropout(0.5),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model on test data
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print('Test accuracy:', test_accuracy)

# # Make predictions
# # Make predictions
# y_pred_prob = model.predict(x_test)
# y_pred = (y_pred_prob > 0.5).astype(int)

# import xgboost as xgb
# from sklearn.metrics import accuracy_score, classification_report

# # Instantiate XGBoost classifier
# xgb_model = xgb.XGBClassifier()

# # Train the model
# xgb_model.fit(x_train, y_train)

# # Make predictions on the test data
# y_pred = xgb_model.predict(x_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Additional evaluation metrics
# print(classification_report(y_test, y_pred))