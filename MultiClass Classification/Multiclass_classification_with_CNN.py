#---------------- MULTICLASS CLASSIFICATION WITH CONVOLUTIONAL NEURAL NETWORK ----------------#
import os
import cv2
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout


#---------------- DATASET ----------------#
'''
The dataset contains images of Multiple fruits for training and testing separate folders! We built a Convolutional Neural
Network for categorising them into four categories, say,  Apple, Banana, Mixed and Orange. The training set contains 
300 images for each class and the testing set contains 60 images of each class. We split up the Testing data into
Testing and Validation dataset having 240 and 60 images respectively.   
Data Source: https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection
'''


#---------------- CREATE TESTING AND TRAINING DATASET ----------------#
# Setting the Dataset Directories
train_path = './dataset/train_zip/train'
test_path = './dataset/test_zip/test'

# Training Data
train_images = []
train_labels = []
shape = (200, 200)
for filename in os.listdir(train_path):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path, filename))
        # Spliting file names and storing the labels for image in list
        train_labels.append(filename.split('_')[0])
        # Resize all images to a specific shape
        img = cv2.resize(img, shape)
        train_images.append(img)
# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values
# Converting train_images to array
train_images = np.array(train_images)
# Splitting Training data into train and validation dataset
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)

# Test Data
test_images = []
test_labels = []
shape = (200, 200)
for filename in os.listdir(test_path):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path, filename))
        # Spliting file names and storing the labels for image in list
        test_labels.append(filename.split('_')[0])
        # Resize all images to a specific shape
        img = cv2.resize(img, shape)
        test_images.append(img)
# Converting test_images to array
test_images = np.array(test_images)
print(len(train_images),len(test_images),len(x_val))


#---------------- VISUALIZE THE DATASET ----------------#
# Visualizing Training data
# Image display in a Grid format
# Dimensions of the plot grid
W_grid = 5
L_grid = 5
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))
axes = axes.ravel() #flaten the 15 x 15 matrix into 225 array

# Generating the Grid
for i in np.arange(0, W_grid * L_grid): #create evenly spaces variables
    image = np.random.randint(0,len(train_images))

    # Converting Labels to Fruit names
    labels = {0: 'apple', 1: 'banana', 2: 'mixed', 3: 'orange'}
    fruit = labels[np.argmax(train_labels[image])]

    # Read and display an image with the selected index
    axes[i].imshow(train_images[image])
    axes[i].text(13,33, fruit, bbox=dict(fill=False, edgecolor='red', linewidth=1))
    axes[i].axis('off')
    if i == 2:
            title = axes[i].set_title('Ground Truth',bbox=dict(fill=False, edgecolor='red', linewidth=1), loc='center', y=1.1)
fig.suptitle('Images of Fruits from Training Data with Labels')
plt.show()


#---------------- CREATING ONVOLUTIONAL NEURAL NETWORK ----------------#
# Creating a Sequential model
model = Sequential()

# Convolutional layer with Pooling layer and Dropout
model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='tanh', input_shape=(200, 200, 3,)))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))

# Flattening the end CNN layer for FCC
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))

# Output Layer
model.add(Dense(4, activation='softmax'))

# Displaying the Convolutional Neural Network Model
model.summary()


#---------------- TRAIN THE MODEL ----------------#
'''
Optimizer : ADAM with learning rate 0.001 and beta 0.90
Loss: Categorical Cross Entropy. We have multiple for the classification task, specifically four different classes of
      fruits.
Mertics: Accuracy. We need to calculates how often predictions equal labels.
'''
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])


# Training the Convolutional Neural Network with the Training dataset
# Check if the model already exists else train from scratch
if os.path.isdir('./Model'):
    # Load the saved model
    keras.models.load_model('Model')
    # Variable to ensure graph plot
    plot_graph = 1
else:
    # Train the model
    history = model.fit(x_train,
                        y_train,
                        epochs = 2,
                        batch_size = 50,
                        validation_data = (x_val,y_val))

    # Save the trained model
    model.save('Model')
    # Variable to ensure graph plot
    plot_graph = 0


#---------------- ACCURACY OF THE MODEL ----------------#
# Adding the Graph for Accuracy in Training and Test Data
if plot_graph == 0:
    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


#---------------- VISUALISING THE RESULT ----------------#
# Image display in a Grid format
# Dimensions of the plot grid
W_grid = 5
L_grid = 5
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))
axes = axes.ravel() #flaten the 15 x 15 matrix into 225 array

# Generating the Grid
for i in np.arange(0, W_grid * L_grid): #create evenly spaces variables
    image = np.random.randint(0, len(test_images))

    # Converting Labels to Fruit names
    labels = {0: 'apple', 1: 'banana', 2: 'mixed', 3: 'orange'}
    actual_fruit = labels[np.argmax(test_labels[image])]

    # Classify the Image according to the Model
    test_image_temp = np.expand_dims(test_images[image], axis=0)
    x = model.predict(np.array(test_image_temp))
    predicted_fruit = labels[np.argmax(x)]

    # Read and display an image with the selected index
    axes[i].imshow(test_images[image])
    axes[i].text(13, 33, actual_fruit, bbox=dict(fill=False, edgecolor='red', linewidth=1))
    axes[i].text(13,93, predicted_fruit, bbox=dict(fill=False, edgecolor='green', linewidth=1))
    axes[i].axis('off')

    if i == 1:
            title = axes[i].set_title('Ground Truth',bbox=dict(fill=False, edgecolor='red', linewidth=1), loc='center', y=1.1)
    if i == 3:
            title = axes[i].set_title('Prediction',bbox=dict(fill=False, edgecolor='green', linewidth=1), loc='center', y=1.1)
fig.suptitle('Images of Cats and Dogs from Testing Data with Ground Truth and Prediction')
plt.show()