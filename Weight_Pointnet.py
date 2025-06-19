#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import open3d as o3d
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras.metrics import RootMeanSquaredError

tf.random.set_seed(1234)


# In[3]:


excel_file_path = r'C:\Users\spaudel6\Desktop\USPLF_Weight\Weights.csv'


df = pd.read_csv(excel_file_path)


df['RFID'] = df['RFID'].astype(str).apply(lambda x: '0' + x if x.isdigit() and len(x) == 15 else x)

# Extract the categories from the 'RFID' column
categories = df['RFID'].tolist()

# Clean the weight values in the '11/1' column, removing "G:" and "lb"
df['weight'] = df['11/29'].str.replace(r"G:|\s+lb", "", regex=True).str.strip()

# Convert 'weight' column to numeric, handling NaN values automatically
df['weight'] = pd.to_numeric(df['weight'], errors="coerce")

# Convert negative labels to absolute positive values
df['weight'] = df['weight'].apply(lambda x: abs(x) if pd.notnull(x) else x)

# Extract cleaned and absolute numeric labels into a list
labels = df['weight'].tolist()

# Create a mapping from category names to labels
category_to_label = dict(zip(categories, labels))

print("Category to label mapping:", category_to_label)


# In[84]:


DIR = r"E:\Last_15_Days\2022-12-01-SPC"


# In[85]:


def parse_dataset(DATA_DIR):
    data_points = []
    data_labels = []
    num_points = 1500
    max_per_category = 50  # Maximum number of point clouds to read per category

    for category in categories:
        category_path = os.path.join(DATA_DIR, str(category))
        print(category_path)
        
        if not os.path.exists(category_path):
            print(f"Category {category} not found, skipping.")
            continue

        file_list = os.listdir(category_path)[:max_per_category]  # Limit to 20 files per category
        for file_name in file_list:
            file_path = os.path.join(category_path, file_name)
            try:
                pcd1 = o3d.io.read_point_cloud(file_path)
                pcd = pcd1.voxel_down_sample(voxel_size=5)
                n_points = np.asarray(pcd.points).shape[0]
                
                # Ensure the point cloud has enough points for sampling
                if n_points < num_points:
                    print(f"Point cloud in {file_path} has fewer than {num_points} points, skipping.")
                    continue

                idx = np.random.choice(n_points, num_points, replace=False)
                pcd_downsampled = np.asarray(pcd.select_by_index(idx).points)

                data_points.append(np.asarray(pcd_downsampled))
                data_labels.append(category_to_label[category])
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    data_points = np.array(data_points)
    data_labels = np.array(data_labels)
    
    # Calculate mean and standard deviation for each coordinate
    x_mean = np.mean(data_points[:, :, 0])
    y_mean = np.mean(data_points[:, :, 1])
    z_mean = np.mean(data_points[:, :, 2])
    x_std = np.std(data_points[:, :, 0])
    y_std = np.std(data_points[:, :, 1])
    z_std = np.std(data_points[:, :, 2])
    
    # Normalize X, Y, and Z coordinates
    data_points[:, :, 0] = (data_points[:, :, 0] - x_mean) / x_std
    data_points[:, :, 1] = (data_points[:, :, 1] - y_mean) / y_std
    data_points[:, :, 2] = (data_points[:, :, 2] - z_mean) / z_std
    
    # Split into training and validation sets
    train_points, val_points, train_labels, val_labels = train_test_split(
        data_points, data_labels, test_size=0.2, random_state=42
    )
    
    return (
        train_points,
        train_labels,
        val_points,
        val_labels
    )


# In[86]:


# def parse_dataset(DATA_DIR):
#     data_points = []
#     data_labels = []
#     num_points = 1500

#     for category in categories:
#         category_path = os.path.join(DATA_DIR, str(category))
#         print(category_path)
        
#         if not os.path.exists(category_path):
#             print(f"Category {category} not found, skipping.")
#             continue

#         for file_name in os.listdir(category_path):
#             file_path = os.path.join(category_path, file_name)
#             pcd1 = o3d.io.read_point_cloud(file_path)
#             pcd = pcd1.voxel_down_sample(voxel_size=5)
#             n_points = np.asarray(pcd.points).shape[0]
#             idx = np.random.choice(n_points, num_points, replace=False)
#             pcd_downsampled = np.asarray(pcd.select_by_index(idx).points)

#             data_points.append(np.asarray(pcd_downsampled))
#             data_labels.append(category_to_label[category])

# #     data_labels = to_categorical(data_labels)
    
#     data_points = np.array(data_points)
    
#     # Calculate mean and standard deviation for each coordinate
#     x_mean = np.mean(data_points[:, :, 0])
#     y_mean = np.mean(data_points[:, :, 1])
#     z_mean = np.mean(data_points[:, :, 2])
#     x_std = np.std(data_points[:, :, 0])
#     y_std = np.std(data_points[:, :, 1])
#     z_std = np.std(data_points[:, :, 2])
    
#     # Normalize X, Y, and Z coordinates
#     data_points[:, :, 0] = (data_points[:, :, 0] - x_mean) / x_std
#     data_points[:, :, 1] = (data_points[:, :, 1] - y_mean) / y_std
#     data_points[:, :, 2] = (data_points[:, :, 2] - z_mean) / z_std
    
#     train_points, val_points, train_labels, val_labels = train_test_split(
#         data_points, data_labels, test_size=0.2, random_state=42
#     )
    
#     return (
#         train_points,
#         train_labels,
#         val_points,
#         val_labels
#     )


# In[87]:


train_points, train_labels, test_points, test_labels = parse_dataset(DIR)


# In[88]:


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.01, 0.01, dtype=tf.float64)
    # shuffle points
    label += tf.random.uniform([], -0.5, 0.50, dtype=tf.float64)
    points = tf.random.shuffle(points)
    return points, label


# In[89]:


print(min(train_labels))


# In[90]:


# num_classes = train_labels.shape[1]
# print(num_classes)

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(32)
# train_dataset = train_dataset.shuffle(len(train_points)).batch(32)
test_dataset = test_dataset.shuffle(len(test_points)).batch(32)


# In[99]:


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.01):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 32)
    x = dense_bn(x, 64)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])
inputs = keras.Input(shape=(1500, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
# x = conv_bn(x, 32)
# x = tnet(x, 32)
# x = conv_bn(x, 32)
# x = conv_bn(x, 64)
# x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 32)
# x = layers.Dropout(0.3)(x)
# x = dense_bn(x, 64)

# x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
# model.load_weights(r"C:\Users\spaudel6\Desktop\USPLF_Weight\Weights\11_1_P5.h5")


# In[100]:


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.9,
    staircase=True)


# In[101]:


model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[RootMeanSquaredError()],
)


# In[102]:


callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=25,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# In[103]:


history = model.fit(train_dataset, epochs=200, validation_data=test_dataset, callbacks=callbacks)


# In[104]:


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
# plt.title('model accuracy')
plt.ylabel('RMSE (lb)')
plt.xlabel('epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('learning_curvepng')


# In[105]:


model.save_weights(r"C:\Users\spaudel6\Desktop\USPLF_Weight\Weights\11_29_P45.h5")


# In[ ]:




