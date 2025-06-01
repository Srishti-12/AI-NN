from keras.datasets import mnist
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

np.random.seed(42)
#mnist.load_data()
print("loading data............")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("**********Data load and tarin test split complted***************")

print ("@@@@@@ Pre Processing Training Data @@@@@@")

print ("******* Adding Gray Scale ***********************")

train_images = train_images.reshape((60000, 28, 28, 1))
#originally in mnist it's (60000,28,28) but for DL  
#-keras have to make to (60000,28,28,1) adding a channel dimension even for gray scale

print("Normalizing on scale 0-1 form 0-255")

train_images = train_images.astype('float32') / 255


print ("@@@@@@ Pre Processing Test Data @@@@@@")

print ("******* Adding Gray Scale ***********************")

test_images = test_images.reshape((10000, 28, 28, 1))
#originally in mnist it's (60000,28,28) but for DL  
#-keras have to make to (60000,28,28,1) adding a channel dimension even for gray scale

print("Normalizing on scale 0-1 form 0-255")

test_images = test_images.astype('float32') / 255

# Convert class vectors (integers) to binary class matrices (one-hot encoding)
#works for both signle nad multi digit and will be useful for categorical cross entropy and softmax activation 


train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)

print("Creating sequential model , craeting instance")

model=models.Sequential()

print ("#### Adding convolutional layers-3 , max pooling, filters, activation function,droppot and dense layers to the CNN model #####")
print ("##############  Forming the CNN Architecture here ####################")
model =Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("********************* MODEL SUMMARY ***************************")
model.summary()

print ("*********************COMPILING the MODEL***********************************")

model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
#Using categorical_crossentropy as it's the best fit loss function for muiti class also works well with softmax
#using softmax as it's not binary data but multilabel data and need probabilitis of each output whixh would then add up to 1
#optimising with adam as it's faster and add up to more optimizers that is RMSProp and ADAGrad.
# I THEN MOVED TO RMSPROP as ADAM was MAKING MY MODEL OVERFIT, after this LOSS IMPROVED FOR VALIODATION and also accuracy slightly

#"training will continue for 5 more epochs after the last time the monitored metric improved
#print("##########Adding early stopping with patience value 5######")
#early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#Implementing early stopping by monitoring validation loss can prevent overfitting. If the validation loss stops
#improving for a set number of epochs (commonly referred to as "patience"), training can be halted early.

print("******************Fitting the MODEL***************************************")
training_history = model.fit(
    train_images, train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    #callbacks=[early_stop] #removed as it was better predicting without this
)

#Training with train_images(X_train), train_labels(Y_train)
#while reserving 10% for validation which will valiadate output after each epoch (our case 50 that means validation 5 times)
#data goes in batches so 64 data elements at a time after which weights will update for the next 64 and so on
#set to a higher epoch otherwise early stop can pause the execution early if the loss stops improving. So setting to higher value


print("Plotting Losses:")

# Set the x-axis and y-axis labels

plt.xlabel('Epoch Number')                    # X-axis will represent epochs

plt.ylabel('Loss')                            # Y-axis will represent loss values

# Plot the training loss
print("training loss")
plt.plot(training_history.history['loss'], label='Training Loss')

# Plot the validation loss
print("validation loss")
plt.plot(training_history.history['val_loss'], label='Validation Loss')

# Add a legend to label the two curves

plt.legend()
# Display the plot

plt.show()

 # 'Plot the model results upon Accuracy:

 # Plot training and validation accuracy over epochs

print("Model result for accuracy")

plt.xlabel('Epoch Number')         # Label for x-axis (number of epochs)

plt.ylabel('Accuracy')             # Label for y-axis (accuracy values)

# Plot training accuracy

plt.plot(training_history.history['accuracy'], label='Training Accuracy')
# Plot validation accuracy

plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')

# Add legend to differentiate between the two curves

plt.legend()

 

# Display the plot

plt.show()

# Evaluate the model on the test data

# This computes the loss and accuracy of the model on the test set

test_loss, test_acc = model.evaluate(test_images, test_labels)
# Print the accuracy on the test set

print(f"Test Accuracy: {test_acc}")

# Save the trained model to a file (mnist.multidigit_recog)

model.save('mnist.multidigit_recog.h5')
