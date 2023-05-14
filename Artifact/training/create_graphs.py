import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model("my_model")
validate = ImageDataGenerator(rescale= 1/255)

labels = ['fist','okay','peace','point','point sideways','rock','stop','thumbs up']

validate_dataset = validate.flow_from_directory('images/validate',target_size=(254,254),
batch_size=16,class_mode="categorical")

y_pred = model.predict(validate_dataset)
rounded_pred = np.argmax(y_pred, axis=-1)

# creating a confusion matrix
matrix = confusion_matrix(y_pred=rounded_pred, y_true=validate_dataset.labels)

ax = sns.heatmap(matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('True Values ')

ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

#Display confusion matrix 
plt.show()

