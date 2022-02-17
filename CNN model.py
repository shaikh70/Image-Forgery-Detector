#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333) 
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))


# In[2]:


from PIL import Image, ImageChops, ImageEnhance
import os
import itertools


# In[3]:


def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image


# In[1]:


real_image_path = r"D:\BTech_Project\archive_4\CASIA2\Tp\Tp_S_NRN_S_O_sec00036_sec00036_00764.tif"
Image.open(real_image_path)


# In[5]:


convert_to_ela_image(real_image_path, 90)


# In[6]:


fake_image_path = r'D:\BTech_Project\archive_4\CASIA2\Tp\Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg'
Image.open(fake_image_path)


# In[7]:


convert_to_ela_image(fake_image_path, 90)


# In[8]:


image_size = (128, 128)


# In[9]:


def prepare_image(image_path):
#     width, height=Image.open(real_image_path).size
#     new_width  = 128
#     new_height = new_width * height / width 
#     new_width  = 128
#     new_height = new_width * height / width 
#     image_size = (int(new_width),int(new_height))
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


# In[10]:


X1 = []
Y1 = []
X2 = []
Y2 = []
X = []
Y = []


# In[11]:


def rando():
    return 0.1


# In[12]:


import random
path = r'D:\BTech_Project\archive_4\CASIA2\Au'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X1.append(prepare_image(full_path))
            Y1.append(1)
            if len(Y1) % 500 == 0:
                print(f'Processing {len(Y1)} images')

random.shuffle(X1,rando)
X1 = X1[:3100]
Y1 = Y1[:3100]
print(len(X1), len(Y1))


# In[13]:


path = r'D:\BTech_Project\archive_4\CASIA2\Tp'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            X2.append(prepare_image(full_path))
            Y2.append(0)
            if len(Y2) % 500 == 0:
                print(f'Processing {len(Y2)} images')
random.shuffle(X2,rando)
X2 = X2[:3000]
Y2 = Y2[:3000]
print(len(X2), len(Y2))


# In[14]:


X1.extend(X2)
Y1.extend(Y2)
X=X1
Y=Y1
print(len(X),len(Y))


# In[15]:


X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)


# In[16]:


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)
X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))


# In[17]:


def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    return model


# In[18]:


model = build_model()
model.summary()


# In[19]:


epochs = 30
batch_size = 32


# In[20]:


init_learning_rate = 1e-4
#optimizer = RMSprop(learning_rate=init_learning_rate, rho=0.9, epsilon=init_learning_rate/epochs, decay=init_learning_rate/50)
#optimizer = SGD(lr=Config.lr, decay=Config.decay, momentum=Config.momentum, nesterov=Config.nesterov)
optimizer = Adam(learning_rate = init_learning_rate, decay = init_learning_rate/epochs)


# In[21]:


model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ['accuracy'])


# In[22]:


early_stopping = EarlyStopping(monitor = 'val_acc',
                              min_delta = 0,
                              patience = 2,
                              verbose = 0,
                              mode = 'auto')


# In[23]:


hist = model.fit(X_train,
                 Y_train,
                 batch_size = batch_size,
                 epochs = epochs,
                validation_data = (X_val, Y_val),
                callbacks = [early_stopping])


# In[24]:


class_names = ['fake', 'real']


# In[25]:


def fun(image_path):
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    return(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')


# In[30]:


print(fun(r"C:\Users\shaik\OneDrive\Desktop\TEST\before-and-after-photoshop-photos-18.jpg"))


# In[27]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:


Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_val,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(2))


# In[ ]:




