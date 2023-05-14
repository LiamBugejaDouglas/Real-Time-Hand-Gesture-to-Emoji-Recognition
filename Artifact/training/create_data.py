import tensorflow as tf
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

train = ImageDataGenerator(rescale= 1/255)
validate = ImageDataGenerator(rescale= 1/255)

#Getting images from repsective directory and spiltting them into two datasets 
train_dataset = train.flow_from_directory('images/test',target_size=(254,254),
batch_size=16,class_mode="categorical")
validate_dataset = validate.flow_from_directory('images/validate',target_size=(254,254),
batch_size=16,class_mode="categorical")

#print(train_dataset.class_indices)
#print(train_dataset.classes)

#Building the CNN using the keras library
model = keras.Sequential(
    [
        #Transfrom the image 
        layers.RandomZoom(0.1),
        layers.RandomRotation(0.1),
        layers.RandomFlip("horizontal", input_shape=(254,254,3)),
        layers.RandomContrast(factor=0.1),

        #Extract the feature map 
        layers.Conv2D(32, (3,3), padding='valid', activation='relu', input_shape=(254,254,3)),
        layers.MaxPool2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPool2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPool2D(2,2),
        
        #Flatten the data
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(128,activation='relu'),
        layers.Dense(8,activation='softmax'),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

model_fit= model.fit(train_dataset, epochs=25, validation_data = validate_dataset) 

model.save("my_model")