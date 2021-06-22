#---------------- SMILING FACES DETECTON WITH CNN ----------------#
import h5py
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


#---------------- DATASET ----------------#
'''
The dataset contains a series of images that can be used to solve the Happy House problem! We need to build a CNN that
can detect smiling faces. The train set has 600 examples. The test set has 150 examples.
Data Source: https://www.kaggle.com/iarunava/happy-house-dataset
'''

# Importing the dataset
filename = 'train_happy.h5'
f = h5py.File(filename, 'r')


#---------------- CREATE TESTING AND TRAINING DATASET ----------------#
# Importing Training set and Test set
happy_training = h5py.File('train_happy.h5', "r")
happy_testing  = h5py.File('test_happy.h5', "r")

X_train = np.array(happy_training["train_set_x"][:]) 
y_train = np.array(happy_training["train_set_y"][:]) 

X_test = np.array(happy_testing["test_set_x"][:])
y_test = np.array(happy_testing["test_set_y"][:])

# Normalising the Data
X_train = X_train/255
X_test = X_test/255


#---------------- VISUALIZE THE DATASET ----------------#
# Random image selection
image = random.randint(1,600)
plt.imshow( X_train[image])
plt.show()

# Image display in a Grid format
# Dimensions of the plot grid
W_grid = 5
L_grid = 5
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))
axes = axes.ravel() #flaten the 15 x 15 matrix into 225 array
n_training = len(X_train) #get the length of the training dataset

# Generating the Grid
for i in np.arange(0, W_grid * L_grid): #create evenly spaces variables
    # Selecting Random image
    index = np.random.randint(0, n_training)
    # Read and display an image with the selected index
    axes[i].imshow( X_train[index])
    axes[i].set_title(y_train[index], fontsize = 25)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)
plt.show()


#---------------- CREATING CONVOLUTIONAL NEURAL NETWORK ----------------#
# Initialise the Convolutional Neural Network
cnn_model = Sequential()

# Convolutional layer with Pooling layer and Dropout
cnn_model.add(Conv2D(64, 6, 6, input_shape = (64,64,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Dropout(0.2))

# Adding another Convulutional layer with Dropout
cnn_model.add(Conv2D(128, 5, 5, activation='relu'))
cnn_model.add(Dropout(0.2))

# Flattening the end CNN layer for FCC
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 128, activation = 'relu'))
cnn_model.add(Dense(units = 1, activation = 'sigmoid'))


#---------------- TRAIN THE MODEL ----------------#
# Compiling the Convolutional Neural Network
'''
Optimizer : ADAM with learning rate 0.001
Loss: Binary Cross Entropy. We have only two classes for the classification task i.e. Smiling or Non-smiling.
Mertics: Accuracy. We need to calculates how often predictions equal labels.
'''
cnn_model.compile(loss ='binary_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

# Training the Convolutional Neural Network with the Training set
epochs = 100
history = cnn_model.fit(X_train,
                        y_train,
                        batch_size = 30,
                        epochs = epochs,
                        verbose = 1)


#---------------- EVALUATE THE MODEL ----------------#
# Evaluate
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))


#---------------- VISUALISING THE RESULT ----------------#
# Predictions for the test data
predicted_classes = cnn_model.predict_classes(X_test)

# Image display in a Grid format
# Dimensions of the plot grid
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Prediction Class = {}\n True Class = {}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)


#---------------- GENERATING THE CONFUSION MATRIX ----------------#
# Generating the Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)
plt.show()