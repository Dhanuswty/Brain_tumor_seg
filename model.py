#from google.colab import auth
#auth.authenticate_user()
#from oauth2client.client import GoogleCredentials
#creds = GoogleCredentials.get_application_default()
import getpass

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add
from keras.models import Model
from keras import regularizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_normal
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import tensorflow as tf

def two_path(X_input):
  
  X = Conv2D(64,(7,7),strides=(1,1),padding='valid')(X_input)
  X = BatchNormalization()(X)
  X1 = Conv2D(64,(7,7),strides=(1,1),padding='valid')(X_input)
  X1 = BatchNormalization()(X1)
  X = layers.Maximum()([X,X1])
  X = Conv2D(64,(4,4),strides=(1,1),padding='valid',activation='relu')(X)
  
  X2 = Conv2D(160,(13,13),strides=(1,1),padding='valid')(X_input)
  X2 = BatchNormalization()(X2)
  X21 = Conv2D(160,(13,13),strides=(1,1),padding='valid')(X_input)
  X21 = BatchNormalization()(X21)
  X2 = layers.Maximum()([X2,X21])
  
  X3 = Conv2D(64,(3,3),strides=(1,1),padding='valid')(X)
  X3 = BatchNormalization()(X3)
  X31 =  Conv2D(64,(3,3),strides=(1,1),padding='valid')(X)
  X31 = BatchNormalization()(X31)
  X = layers.Maximum()([X3,X31])
  X = Conv2D(64,(2,2),strides=(1,1),padding='valid',activation='relu')(X)
  
  X = Concatenate()([X2,X])
  #X = Conv2D(5,(21,21),strides=(1,1))(X)
  #X = Activation('softmax')(X)
  
  #model = Model(inputs = X_input, outputs = X)
  return X

def input_cascade(input_shape1,input_shape2):
  
  X1_input = Input(input_shape1)
  X1 = two_path(X1_input)
  X1 = Conv2D(5,(21,21),strides=(1,1),padding='valid',activation='relu')(X1)
  X1 = BatchNormalization()(X1)
  
  X2_input = Input(input_shape2)
  X2_input1 = Concatenate()([X1,X2_input])
  #X2_input1 = Input(tensor = X2_input1)
  X2 = two_path(X2_input1)
  X2 = Conv2D(5,(21,21),strides=(1,1),padding='valid')(X2)
  X2 = BatchNormalization()(X2)
  X2 = Activation('softmax')(X2)
  
  model = Model(inputs=[X1_input,X2_input],outputs=X2)
  return model

def MFCcascade(input_shape1,input_shape2):
  
  X1_input = Input(input_shape1)
  X1 = two_path(X1_input)
  X1 = Conv2D(5,(21,21),strides=(1,1),padding='valid',activation='relu')(X1)
  X1 = BatchNormalization()(X1)
  #X1 = MaxPooling2D((2,2))(X1)
  
  X2_input = Input(input_shape2)
  X2 = two_path(X2_input)
  
  X2 = Concatenate()([X1,X2])
  X2 = Conv2D(5,(21,21),strides=(1,1),padding='valid',activation='relu')(X2)
  X2 = BatchNormalization()(X2)
  X2 = Activation('softmax')(X2)
  
  model = Model(inputs=[X1_input,X2_input],outputs=X2)
  return model

m = MFCcascade((53,53,4),(33,33,4))
#m.summary()

m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
m.save('trial_0001_MFCcascade_acc.h5')
m1 = input_cascade((65,65,4),(33,33,4))
#m1.summary()

import os
#os.chdir('drive/brat')

def model_gen(input_dim,x,y,slice_no):
  X1 = []
  X2 = []
  Y = []
  print(int((input_dim)/2))
  for i in range(int((input_dim)/2),175-int((input_dim)/2)):
    for j in range(int((input_dim)/2),195-int((input_dim)/2)):
      X2.append(x[i-16:i+17,j-16:j+17,:])
      X1.append(x[i-int((input_dim)/2):i+int((input_dim)/2)+1,j-int((input_dim)/2):j+int((input_dim)/2)+1,:])
      Y.append(y[i,slice_no,j])
      
  X1 = np.asarray(X1)
  X2 = np.asarray(X2)
  Y = np.asarray(Y)
  d = [X1,X2,Y]
  return d

import SimpleITK as sitk
import numpy as np

def data_gen(path,slice_no,model_no):
  p = os.listdir(path)
  p.sort(key=str.lower)
  arr = []
  for i in range(len(p)):
    if(i != 4):
      p1 = os.listdir(path+'/'+p[i])
      p1.sort()
      img = sitk.ReadImage(path+'/'+p[i]+'/'+p1[-1])
      arr.append(sitk.GetArrayFromImage(img))
    else:
      p1 = os.listdir(path+'/'+p[i])
      img = sitk.ReadImage(path+'/'+p[i]+'/'+p1[0])
      y = sitk.GetArrayFromImage(img)    
  data = np.zeros((196,176,216,4))
  for i in range(196):
    data[i,:,:,0] = arr[0][:,i,:]
    data[i,:,:,1] = arr[1][:,i,:]
    data[i,:,:,2] = arr[2][:,i,:]
    data[i,:,:,3] = arr[3][:,i,:]
  x = data[slice_no]
  
  if(model_no == 0):
    X1 = []
    for i in range(16,159):
      for j in range(16,199):
        X1.append(x[i-16:i+17,j-16:j+17,:])
    Y1 = []
    for i in range(16,159):
      for j in range(16,199):
        Y1.append(y[i,slice_no,j]) 
    X1 = np.asarray(X1)
    Y1 = np.asarray(Y1)
    d = [X1,Y1]
  elif(model_no == 1):
    d = model_gen(65,x,y,slice_no)
  elif(model_no == 2):
    d = model_gen(56,x,y,slice_no)
  elif(model_no == 3):
    d = model_gen(53,x,y,slice_no)  
    
  return d

d = data_gen('/home/sweety/Desktop/Brain_tumor_seg/BRATS-2/Image_Data/LG/0001',100,3)

d[2].all == 0
len(d[0])


import SimpleITK as sitk
import numpy as np

y = np.zeros((17589,1,1,5))

for i in range(y.shape[0]):
  y[i,:,:,d[2][i]] = 1
sample = np.zeros((5,1))
for i in range(5):
  sample[i] = np.sum(y[:,:,:,i])
#print(sample/np.sum(sample))

from sklearn.metrics import f1_score

X1 = np.asarray(d[0])
X1.shape
X2 = np.asarray(d[1])
X2.shape
m1.inputs
#m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[f1_score])
m_info = m.fit([X1,X2],y,epochs=20,batch_size=256)
m.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(d[2]),
                                                 d[2])
class_weights
import keras
model = keras.models.load_model('trial_0001_MFCcas_dim2_128_acc.h5')
m_info = m.fit([X1,X2],y,epochs= 20,batch_size = 256,class_weight = class_weights)

import matplotlib.pyplot as plt
plt.plot(m_info.history['acc'])
#plt.plot(m_info.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

m.save('trial_MFCcascade_acc.h5')
model.evaluate([X1,X2],y,batch_size = 1024)

model_info = model.fit([X1,X2],y,epochs=30,batch_size=256,class_weight= class_weights)

import matplotlib.pyplot as plt
plt.plot(model_info.history['acc'])
#plt.plot(m_info.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('trial_0001_MFCcas_dim2_128_acc.h5')
model.evaluate([X1,X2],y,batch_size = 1024)
pred = model.predict([X1,X2],batch_size = 1024)
pred = np.around(pred)
pred1 = np.dot(pred.reshape(17589,5),np.array([0,1,2,3,4]))
y1 = np.dot(y.reshape(17589,5),np.array([0,1,2,3,4]))

y2 = np.argmax(y.reshape(17589,5),axis = 1)
y2.all() == 0
y1.all()==0
from sklearn import metrics
f1 = metrics.f1_score(y1,pred1,average='micro')
f1


p1 = metrics.precision_score(y1,pred1,average='micro')
p1

r1 = metrics.recall_score(y1,pred1,average='micro')
r1

p2 = metrics.precision_score(y1,pred2,average='micro')
p2

from sklearn.utils import class_weight


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(d[2]),
                                                 d[2])


class_weights


m1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
m1_info = m1.fit([X1,X2],y,epochs=20,batch_size=256,class_weight= class_weights)
import matplotlib.pyplot as plt
plt.plot(m1_info.history['acc'])
#plt.plot(m_info.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
m1.save('trial_0001_input_cascade_acc.h5')

