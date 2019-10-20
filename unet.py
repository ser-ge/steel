#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
import pickle


# In[6]:




# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


# In[7]:


testdir ='data/test_images/'
trainddir = 'data/train_images/'


# In[8]:



if tf.test.is_gpu_available(cuda_only=True):
    print('Found GPU')
else:
    print('No GPU!')
        
    


# In[9]:


train_df=pd.read_csv('data/train.csv').fillna(-1)
train_df['ImageId'] = train_df.ImageId_ClassId.apply(lambda row : row.split('_')[0])
train_df['ClassId'] = train_df.ImageId_ClassId.apply(lambda row : row.split('_')[1])


# In[10]:


train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis = 1)
grouped_EncodedPixels= train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)


# In[11]:


def rle_to_mask(rle_string,height,width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rleString (str): Description of arg1 
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img


# In[12]:


def mask_to_rle(mask):
    '''
    Convert a mask into RLE
    
    Parameters: 
    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background

    Returns: 
    sring: run length encoding 
    '''
    pixels= mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[13]:


sns.countplot(train_df[train_df.EncodedPixels!=-1]['ClassId'])


# In[14]:


num_defects = grouped_EncodedPixels.apply(lambda x : sum([True for i in x if i[1]!=-1]) )
sns.countplot(num_defects)


# In[15]:


img_path = os.path.join(trainddir,grouped_EncodedPixels.index[0])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = rle_to_mask(grouped_EncodedPixels[0][0][1], img.shape[0], img.shape[1])


# In[16]:


im_m = cv2.bitwise_or(img, mask)


# In[17]:


plt.imshow(im_m, cmap='gray')


# In[18]:


class DataGenerator(tf.keras.utils.Sequence):
   
   def __init__(self, list_ids, labels, image_dir, batch_size=32, img_h=256,img_w=512,shuffle=True, channels=1 ):
       self.list_ids = list_ids
       self.labels = labels
       self.image_dir = image_dir
       self.batch_size = batch_size
       self.img_h = img_h
       self.img_w = img_w
       self.channels = channels
       self.shuffle = shuffle
       self.on_epoch_end()

   def __len__(self):
       return int(np.floor(len(self.list_ids)) / self.batch_size)
   
   def on_epoch_end(self):
       self.indexes = np.arange(len(self.list_ids))
       if self.shuffle:
           np.random.shuffle(self.indexes)

   def __data_generation(self, list_ids_temp):

       X = np.empty((self.batch_size, self.img_h, self.img_w, self.channels))
       y = np.empty((self.batch_size, self.img_h, self.img_w, 4))

       for i, Id in enumerate(list_ids_temp):

           path = os.path.join(self.image_dir, Id)
           image = cv2.imread(path,0)
           image_resized = cv2.resize(image, (self.img_w,self.img_h))
           image_resized = np.array(image_resized, dtype=np.float32)

           #             normalise
           image_resized -= image_resized.mean()
           image_resized /= image_resized.std()

           mask = np.empty((self.img_h,self.img_w,4))

           height_org, width_org = image.shape

           for j, img_class in enumerate(['1','2','3','4']):

               rle = self.labels.get(Id + '_' + img_class)

               if rle is None:
                   class_mask = np.zeros((width_org,height_org))
               else:
                   class_mask = rle_to_mask(rle,height_org, width_org)

               class_mask_resized = cv2.resize(class_mask, (self.img_w,self.img_h))

               mask[... , j] = class_mask_resized

           X[i,] = np.expand_dims(image_resized, axis=2)
           y[i,] = mask

       y = (y > 0).astype(int)

       return X , y
       
   def __getitem__(self, index):
       indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
       # get list of IDs
       list_ids_temp = [self.list_ids[k] for k in indexes]
       # generate data


       X, y = self.__data_generation(list_ids_temp)
       return X, y                          


# In[19]:


masks = {}
for index, row in train_df[train_df['EncodedPixels']!=-1].iterrows():
    masks[row['ImageId_ClassId']] = row['EncodedPixels']


# In[20]:


train_image_ids = train_df['ImageId'].unique()


# In[21]:


img_h = int(256)
img_w = 800
params ={'img_h' : img_h,
        'img_w': img_w,
        'batch_size': 12,
        'image_dir':trainddir,
        'shuffle':True}


# In[22]:


X_train, X_val = train_test_split(train_image_ids,test_size=0.1, random_state=42)

val_gen = DataGenerator(X_val,masks, **params)
train_gen = DataGenerator(X_train,masks, **params)


# In[23]:


X, y = train_gen.__getitem__(2)
    


# In[24]:


print('The shape of X is {}'.format(X.shape))
print('The shape of y is {}'.format(y.shape))


# In[25]:


# Unet: https://arxiv.org/pdf/1505.04597.pdf

def conv_block(x, filters, max_pool=True):
    
    if max_pool:
        x=MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(filters=filters,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(filters=filters,kernel_size=3, activation='relu', padding='same')(x)
    
    return x

def conv_concatanate(x, shortcut, filters):
    
    x = Concatenate()([x,shortcut])
    x = conv_block(x, filters=filters, max_pool=False)
    return x

def up_conv_block(x, shortcut, filters):
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(filters=filters,kernel_size=2, activation='relu', padding='same')(x)
    x = conv_concatanate(x,shortcut, filters)
    return x

def conv_out(x, output_filters):
    x = Conv2D(filters=output_filters,kernel_size=1, activation='sigmoid', padding='same')(x)
    return x


def unet(input_size,num_output_masks,init_filters=64):

    x = Input(input_size)
    
    conv_down1 = conv_block(x,init_filters, max_pool=False)
    
    conv_down2 = conv_block(conv_down1, init_filters*2)
    conv_down3 = conv_block(conv_down2, init_filters*4)
    conv_down4 = conv_block(conv_down3, init_filters*6)
    
    conv_down5 = conv_block(conv_down4, init_filters*8)
    
    conv_up1 = up_conv_block(conv_down5, conv_down4,init_filters*6)
    conv_up2 = up_conv_block(conv_up1, conv_down3,init_filters*4)
    conv_up3 = up_conv_block(conv_up2, conv_down2,init_filters*2)
    
    conv_up4 = up_conv_block(conv_up3, conv_down1,init_filters)
    
    output = conv_out(conv_up4,num_output_masks)
    
    model = Model(x, output)
    
    return model 


# In[26]:


stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "cp-{epoch}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

save= tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=True, save_freq='epoch')

model = unet((img_h,img_w,1), num_output_masks=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss = binary_crossentropy
model.compile(optimizer=optimizer,loss=loss, metrics=['binary_crossentropy'])

print('model compiled')


# In[27]:


print('begining training')

history= model.fit_generator(generator=train_gen, epochs=30, verbose=True, validation_data=val_gen, callbacks=[stop,save])

print('training complete')
model.save('UnetSteel_1.h5')
print('model saved')
with open('unet_history_pickle','wb') as f:
    pickle.dump(history,f)
print('history saved')


# In[ ]:





# In[ ]:




