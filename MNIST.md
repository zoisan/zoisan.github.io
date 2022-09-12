# MNIST Dataset


```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics  # too many functions to import, so import the whole module
from sklearn.datasets import load_digits

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

%matplotlib inline

import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
```


```python
BASE_DIR = "."
DEBUG = False

np.random.seed(42)
```

## Load Dataset


```python
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```


```python
X_train.shape, X_test.shape
```




    ((60000, 28, 28), (10000, 28, 28))




```python
n_features = X_train.shape[1]*X_train.shape[2]
n_features
```




    784




```python
Y_train
```




    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)




```python
np.unique(Y_train)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)




```python
n_digits = len(np.unique(Y_train))
n_digits
```




    10




```python
df_Y = pd.DataFrame(Y_train, columns=["target"])
df_Y.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>60000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.453933</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.889270</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_Y_counts = df_Y.groupby('target')
df_Y_counts.size()
```




    target
    0    5923
    1    6742
    2    5958
    3    6131
    4    5842
    5    5421
    6    5918
    7    6265
    8    5851
    9    5949
    dtype: int64



## Plot first few images


```python
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
```


    
![png](output_13_0.png)
    


## Prepare the Data

### Normailize data


```python
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
```

### Encoding the target values


```python
# Convert 0-9 digits (Y-train) into categorical variables
Y_mapped = np.array([[0]*n_digits for _ in range(Y_train.shape[0])])
for i in range(len(Y_train)):
    Y_mapped[i][Y_train[i]] = 1

Y_train = Y_mapped
Y_train
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 1, 0]])




```python
# Convert 0-9 digits (Y-test) into categorical variables
Y_test_mapped = np.array([[0]*n_digits for _ in range(Y_test.shape[0])])
for i in range(len(Y_test)):
    Y_test_mapped[i][Y_test[i]] = 1

Y_test = Y_test_mapped
Y_test
```




    array([[0, 0, 0, ..., 1, 0, 0],
           [0, 0, 1, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])



### Split the data into training, validation and test sets



```python
train_size = 50000
valid_size = 10000
test_size = 10000
```


```python
X_train = X_train[:train_size]
X_valid = X_train[-valid_size:]

Y_train = Y_train[:train_size]
Y_valid = Y_train[-valid_size:]
```

# Simple Feedforward Neural Network (FNN)

Let's first build a Sequential Deep Learning Model


```python
# create model
fnn_model = Sequential()

#Flatten the input values, this will be the input layer
fnn_model.add(Flatten(input_shape=(28, 28)))

#first hidden layer
fnn_model.add(Dense(n_features, kernel_initializer='normal', activation='relu'))

#second hidden layer
fnn_model.add(Dense(128, kernel_initializer='normal', activation='relu'))

#output layer
fnn_model.add(Dense(n_digits, kernel_initializer='normal', activation='softmax'))

# print a summary and check if we created the network intended
fnn_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 784)               615440    
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 717,210
    Trainable params: 717,210
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Compile model
fnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
fnn_model_history = fnn_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10, batch_size=100, verbose=1)

# Final evaluation of the model
fnn_model_loss, fnn_model_acc = fnn_model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Test Accuracy={0:.2f}%   (categorical_crossentropy) loss={1:.2f}".format(fnn_model_acc*100, fnn_model_loss))
```

    Epoch 1/10
    500/500 [==============================] - 2s 4ms/step - loss: 0.3133 - accuracy: 0.9102 - val_loss: 0.1469 - val_accuracy: 0.9558
    Epoch 2/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.1097 - accuracy: 0.9668 - val_loss: 0.0768 - val_accuracy: 0.9776
    Epoch 3/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0665 - accuracy: 0.9798 - val_loss: 0.0401 - val_accuracy: 0.9880
    Epoch 4/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0447 - accuracy: 0.9860 - val_loss: 0.0316 - val_accuracy: 0.9902
    Epoch 5/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0311 - accuracy: 0.9902 - val_loss: 0.0227 - val_accuracy: 0.9936
    Epoch 6/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0208 - accuracy: 0.9936 - val_loss: 0.0164 - val_accuracy: 0.9956
    Epoch 7/10
    500/500 [==============================] - 2s 4ms/step - loss: 0.0152 - accuracy: 0.9952 - val_loss: 0.0114 - val_accuracy: 0.9963
    Epoch 8/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0125 - accuracy: 0.9960 - val_loss: 0.0149 - val_accuracy: 0.9952
    Epoch 9/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0117 - accuracy: 0.9961 - val_loss: 0.0150 - val_accuracy: 0.9956
    Epoch 10/10
    500/500 [==============================] - 2s 3ms/step - loss: 0.0089 - accuracy: 0.9972 - val_loss: 0.0088 - val_accuracy: 0.9971
    Baseline Test Accuracy=97.54%   (categorical_crossentropy) loss=0.11
    

## Plot the accuracy and the loss


```python
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ax[0].plot(fnn_model_history.history['accuracy'])
ax[0].plot(fnn_model_history.history['val_accuracy'])

ax[0].set(ylabel='accuracy')
ax[0].legend(['train', 'test'], loc=(1.02, -0.1), fontsize=15)


# summarize history for loss
ax[1].plot(fnn_model_history.history['loss'])
ax[1].plot(fnn_model_history.history['val_loss'])

ax[1].set(xlabel='Epoch number', ylabel='loss')

fig.tight_layout()

plt.title('Progress of model accuracy and loss', y=2.2, fontsize=20)

plt.show()
```


    
![png](output_27_0.png)
    


# Convolutional Neural Network (CNN)

Let's now build a simple Convolutional Neural Network model


```python
n_width, n_length = X_train.shape[1:]
n_width, n_length
```




    (28, 28)




```python
# We need to convert the input into (samples, rows, cols, channels) format
X_train = X_train.reshape(X_train.shape[0], n_width, n_length, 1).astype('float32')
X_valid = X_valid.reshape(X_valid.shape[0], n_width, n_length, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], n_width, n_length, 1).astype('float32')
X_train.shape, X_valid.shape, X_test.shape
```




    ((50000, 28, 28, 1), (10000, 28, 28, 1), (10000, 28, 28, 1))




```python
X_train.shape[1:]
```




    (28, 28, 1)




```python
#create the model
cnn_model = Sequential()

# 1st Convolution layer
cnn_model.add(Conv2D(64, (3,3), input_shape=X_train.shape[1:], activation='relu'))

#max-pooling
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout to prevent overfitting
cnn_model.add(Dropout(0.5))

#Flatten (this is necessary before Dense layers)
cnn_model.add(Flatten())

#Dense Layer
cnn_model.add(Dense(128, activation='relu'))

#Output Layer
cnn_model.add(Dense(n_digits, activation='softmax'))

#print the summary of the model created
cnn_model.summary()

```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 64)        640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 13, 13, 64)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 10816)             0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 128)               1384576   
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 1,386,506
    Trainable params: 1,386,506
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compile the model
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
cnn_model_history = cnn_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10, batch_size=100, verbose=2)

#Evaluate the model
cnn_model_loss, cnn_model_acc = cnn_model.evaluate(X_test, Y_test, verbose=0) 
print(f"Baseline accuracy is {cnn_model_acc:.2f}% and (categorical crossentropy) loss is {cnn_model_loss:.5f}.")
```

    Epoch 1/10
    500/500 - 16s - loss: 0.3041 - accuracy: 0.9105 - val_loss: 0.1290 - val_accuracy: 0.9622
    Epoch 2/10
    500/500 - 15s - loss: 0.1192 - accuracy: 0.9641 - val_loss: 0.0788 - val_accuracy: 0.9750
    Epoch 3/10
    500/500 - 15s - loss: 0.0853 - accuracy: 0.9736 - val_loss: 0.0512 - val_accuracy: 0.9842
    Epoch 4/10
    500/500 - 15s - loss: 0.0671 - accuracy: 0.9789 - val_loss: 0.0397 - val_accuracy: 0.9878
    Epoch 5/10
    500/500 - 16s - loss: 0.0570 - accuracy: 0.9821 - val_loss: 0.0305 - val_accuracy: 0.9917
    Epoch 6/10
    500/500 - 15s - loss: 0.0460 - accuracy: 0.9849 - val_loss: 0.0349 - val_accuracy: 0.9879
    Epoch 7/10
    500/500 - 15s - loss: 0.0395 - accuracy: 0.9868 - val_loss: 0.0216 - val_accuracy: 0.9934
    Epoch 8/10
    500/500 - 15s - loss: 0.0349 - accuracy: 0.9882 - val_loss: 0.0149 - val_accuracy: 0.9963
    Epoch 9/10
    500/500 - 15s - loss: 0.0321 - accuracy: 0.9892 - val_loss: 0.0153 - val_accuracy: 0.9955
    Epoch 10/10
    500/500 - 15s - loss: 0.0262 - accuracy: 0.9913 - val_loss: 0.0095 - val_accuracy: 0.9982
    Baseline accuracy is 0.98% and (categorical crossentropy) loss is 0.05515.
    

### Let's Plot the accuracy and the loss


```python
fig, ax = plt.subplots(2, 1, figsize=(10,8))

#summarize the history for accuracy
ax[0].plot(cnn_model_history.history['accuracy'])
ax[0].plot(cnn_model_history.history['val_accuracy'])

ax[0].set(ylabel='Accuracy')
ax[0].legend(['train','test'], loc=(1.1,-0.1), fontsize=18)

#summarize the history for loss
ax[1].plot(cnn_model_history.history['loss'])
ax[1].plot(cnn_model_history.history['val_loss'])

ax[1].set(xlabel="Epoch number", ylabel='Loss')

fig.tight_layout()
plt.title("Progress of Model Accuracy and Loss", y=2.2, fontsize=18)

plt.show()
```


    
![png](output_35_0.png)
    


## Computing predictive performance


```python
Y_test_pred_class = np.argmax(cnn_model.predict(X_test), axis=-1)
```


```python
Y_test_pred_class
```




    array([7, 2, 1, ..., 4, 5, 6], dtype=int64)




```python
Y_test_pred_class_probabilities = cnn_model.predict(X_test)
Y_test_pred_class_probabilities
```




    array([[4.21717550e-09, 3.98691242e-07, 1.30994385e-06, ...,
            9.99970675e-01, 5.94761262e-09, 8.90100296e-08],
           [1.10744975e-06, 3.71877977e-05, 9.99926329e-01, ...,
            1.37821712e-10, 1.06311612e-10, 1.74608409e-15],
           [1.20732977e-07, 9.99983907e-01, 3.17484994e-09, ...,
            5.11700406e-08, 1.55998896e-05, 5.66299985e-09],
           ...,
           [4.86830437e-11, 3.85884080e-09, 1.51658146e-12, ...,
            3.68935127e-09, 2.59681059e-07, 7.23217454e-07],
           [8.34951308e-08, 1.30255640e-09, 1.70988046e-09, ...,
            1.12565335e-08, 6.81336503e-03, 7.47073792e-11],
           [2.43834592e-07, 1.26413369e-09, 1.27645521e-07, ...,
            9.03012149e-13, 3.79992429e-08, 1.48009341e-10]], dtype=float32)



Before we can evaluate the performance we also need to convert our one-hot encoded Y_test into its class:



```python
Y_test_class = np.array([np.argmax(y, axis=None, out=None) for y in Y_test])
Y_test_class
```




    array([7, 2, 1, ..., 4, 5, 6], dtype=int64)



### Confusion Matrix


```python
print(metrics.confusion_matrix(Y_test_class, Y_test_pred_class))
```

    [[ 973    0    0    1    0    0    3    0    2    1]
     [   1 1122    2    3    0    0    4    0    3    0]
     [   0    1 1011    8    0    0    2    6    3    1]
     [   0    0    0 1003    0    3    0    1    3    0]
     [   0    0    0    0  975    0    1    0    2    4]
     [   1    0    1   10    0  874    4    0    2    0]
     [   5    2    0    0    2    2  944    0    3    0]
     [   0    4   10    6    0    0    0 1004    3    1]
     [   4    0    1    2    1    0    1    2  960    3]
     [   3    5    0    8   11    1    0    1    4  976]]
    

#### Classification Report


```python
print(metrics.classification_report(Y_test_class, Y_test_pred_class))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99       980
               1       0.99      0.99      0.99      1135
               2       0.99      0.98      0.98      1032
               3       0.96      0.99      0.98      1010
               4       0.99      0.99      0.99       982
               5       0.99      0.98      0.99       892
               6       0.98      0.99      0.98       958
               7       0.99      0.98      0.98      1028
               8       0.97      0.99      0.98       974
               9       0.99      0.97      0.98      1009
    
        accuracy                           0.98     10000
       macro avg       0.98      0.98      0.98     10000
    weighted avg       0.98      0.98      0.98     10000
    
    

## ROC Curve and AUROC Values


```python
# The ROC curve and the area under the ROC curve for each class
fpr = [0.0]*n_digits #false positive
tpr = [0.0]*n_digits #true positive
roc_auc = [0.0]*n_digits

for i in range(n_digits):
    fpr[i], tpr[i], _ = metrics.roc_curve(Y_test[:, i], Y_test_pred_class_probabilities[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
```


```python
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gold', 'maroon', 'orange', 'purple']
```


```python
plt.figure(figsize=(16, 12))
for i, color in zip(range(n_digits), colors):
    plt.plot(fpr[i], tpr[i], lw=2, color=color,
             label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.axis("tight")
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC (Receiver Operating Characteristic) curves of digits', fontsize=25, position=(0.5,1.03))
plt.legend(loc=(0.55, 0.01), fontsize=14)
plt.show()
```


    
![png](output_49_0.png)
    


## Saving models

Everything looks good so far. We can now save the models.

Saving the models into single files:


```python
fnn_model.save("fnn_model.h5")
cnn_model.save("cnn_model.h5")
```


```python
# save model architecture as JSON and weights as HDF5:
cnn_model_json_str_saved = cnn_model.to_json()
with open('cnn_model_architecture.json', 'w') as f:
    f.write(cnn_model_json_str_saved)

cnn_model.save_weights("cnn_model_weights.h5")
```
