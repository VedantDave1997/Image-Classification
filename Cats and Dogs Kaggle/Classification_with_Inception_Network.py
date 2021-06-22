#---------------- CATS AND DOGS CLASSIFICATION WITH INCEPTION NET ----------------#
import os
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import applications
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D


#---------------- DATASET ----------------#
'''
The dataset contains images of Dogs and Cats for training and testing separate folders! We use the Inception Net for 
categorising them into Dogs and Cats. The training set contains 4000 images for each class and the testing set
contains 1000 images of each class.
Data Source: https://www.kaggle.com/chetankv/dogs-cats-images
'''
# Dataset Parameters
image_width, image_height = 224,224
nb_train_samples = 400
nb_test_samples = 100
epochs = 15
batch_size = 10


#---------------- CREATE TESTING AND TRAINING DATASET ----------------#
# Setting the Dataset Directories
training_set_directory = "./dataset/training_set"
test_set_directory = "./dataset/test_set"

# Creating Image Data Genetator
# Training Data Generator
training_data_generator = ImageDataGenerator(rescale=1./255,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True
                                             )

# Test Data Generator
test_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = training_data_generator.flow_from_directory(training_set_directory,
                                                              target_size=(image_width, image_height),
                                                              batch_size=batch_size,
                                                              class_mode='binary'
                                                              )

test_generator = test_data_generator.flow_from_directory(test_set_directory,
                                                         target_size=(image_width, image_height),
                                                         batch_size=batch_size,
                                                         class_mode='binary'
                                                         )


#---------------- VISUALIZE THE DATASET ----------------#
# Image display in a Grid format
# Dimensions of the plot grid
W_grid = 5
L_grid = 5
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25))
axes = axes.ravel() #flaten the 15 x 15 matrix into 225 array

# Generating the Grid
for i in np.arange(0, W_grid * L_grid): #create evenly spaces variables
    img, label = train_generator.next()
    img_train = img[0]
    img_train_label = label[0]

    # Converting Labels to Animal names
    animal = 'Cat' if img_train_label == 0 else 'Dog'

    # Read and display an image with the selected index
    axes[i].imshow(img_train)
    axes[i].text(13,33, animal, bbox=dict(fill=False, edgecolor='red', linewidth=1))
    axes[i].axis('off')
    if i == 2:
            title = axes[i].set_title('Ground Truth',bbox=dict(fill=False, edgecolor='red', linewidth=1), loc='center', y=1.1)
fig.suptitle('Images of Cats and Dogs from Training Data with Labels')
plt.show()



#---------------- CREATING INCEPTION NETWORK ----------------#
# Creating Base Model for Inception Network
base_model= applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(image_width, image_height,3))

# Defining Inception Network Parameters
inet_model=Sequential()
inet_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:],data_format=None))
inet_model.add(Dense(256,activation='relu'))
inet_model.add(Dropout(0.5))
inet_model.add(Dense(1,activation='sigmoid'))

# Defining and Displaying Network Model
model= Model(inputs=base_model.input, outputs=inet_model(base_model.output))
model.summary()


#---------------- TRAIN THE MODEL ----------------#
'''
Optimizer : ADAM with learning rate 0.001 and beta 0.90
Loss: Binary Cross Entropy. We have only two classes for the classification task i.e. Cats or Dogs.
Mertics: Accuracy. We need to calculates how often predictions equal labels.
'''
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# Training the Inception Network with the Training dataset
# Check if the model already exists else train from scratch
if os.path.isdir('./Model'):
    # Load the saved model
    keras.models.load_model('Model')
else:
    # Train the model
    history = model.fit_generator(train_generator,
                                  epochs = epochs,
                                  steps_per_epoch = nb_train_samples // batch_size,
                                  validation_data = test_generator,
                                  validation_steps = nb_test_samples // batch_size)
    # Save the trained model
    model.save('Model')


#---------------- ACCURACY OF THE MODEL ----------------#
# Adding the Graph for Accuracy in Training and Test Data
if not os.path.isdir('./Model'):
    plt.figure()
    plt.plot(history.history['accuracy'], 'RED', label = 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'BLUE', label = ' Testing Accuracy')
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
    img, label = test_generator.next()
    img_test = img[0]
    img_test_label = label[0]

    # Classify the Image according to the Model
    img_test1 = image.img_to_array(img_test)
    x = np.expand_dims(img_test1, axis=0)
    probability = model.predict(x)
    actual_animal = 'Cat' if img_test_label == 0 else 'Dog'
    predicted_animal = 'Cat' if probability < 0.5 else 'Dog'

    # Read and display an image with the selected index
    axes[i].imshow(img_test)
    axes[i].text(13,33, actual_animal, bbox=dict(fill=False, edgecolor='red', linewidth=1))
    axes[i].text(13,93, predicted_animal, bbox=dict(fill=False, edgecolor='green', linewidth=1))
    axes[i].axis('off')

    if i == 1:
            title = axes[i].set_title('Ground Truth',bbox=dict(fill=False, edgecolor='red', linewidth=1), loc='center', y=1.1)
    if i == 3:
            title = axes[i].set_title('Prediction',bbox=dict(fill=False, edgecolor='green', linewidth=1), loc='center', y=1.1)
fig.suptitle('Images of Cats and Dogs from Testing Data with Ground Truth and Prediction')
plt.show()