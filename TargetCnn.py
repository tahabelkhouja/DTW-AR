

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:57:34 2019

@author: BkTaha
"""

import os
import copy


import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import pickle as pkl
from tqdm import tqdm
from dtw_master import dtw as dtwM

from DTWCostMatrixFn import path_conversion, dtw_random_path


def dtw_differntiable(path, x, y, tf_norm=2):
    """
    Make the optimal path a distance function
    """
    x_path = tf.convert_to_tensor(path[0])
    y_path = tf.convert_to_tensor(path[1])   
    if len(x_path) != len(y_path):
        raise ValueError("Error in DTW path length") 
    else:
        dtw_dist = tf.norm(x[x_path[0]] - y[y_path[0]], ord=tf_norm)
        for i in range(1, len(x_path)):
            dtw_dist = tf.add(dtw_dist, tf.norm(x[x_path[i]] - y[y_path[i]], ord=tf_norm))
    return dtw_dist
    
#CNN Architecture
class cnn_class():
    def __init__(self, name, seg_size, channel_nb, class_nb, arch='1'):
        self.name = name
        self.seg_size = seg_size
        self.channel_nb = channel_nb
        self.class_nb = class_nb
        self.x_holder = []
        self.y_holder = []
        self.y_ =[]
        
        
        if arch=='0':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(20,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=2),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='1':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(66,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=4),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='2':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(100,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 5], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 3], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=1),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation=tf.nn.relu),
                tf.keras.layers.Dense(100, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='3':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(100,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 6], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(25,[1, 3], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=1),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation=tf.nn.relu),
                tf.keras.layers.Dense(50, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
                
        elif arch=='lstm':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Reshape((self.seg_size, self.channel_nb)),
                tf.keras.layers.LSTM(128, activation="relu", return_sequences=True),
                tf.keras.layers.LSTM(64,  activation="relu", return_sequences=False),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='transformer':
            pass

        else:
            raise NotImplementedError("Architecture Not Implemented")

        if arch=='transformer':
            self.trunk_model, self.model = self.get_transformer_model(input_shape=(1, self.seg_size, self.channel_nb), class_nb=self.class_nb)
        else:
            self.model = tf.keras.Sequential([self.trunk_model,
                tf.keras.layers.Softmax()])

        #Training Functions
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)


    def get_transformer_model(self, input_shape, class_nb,# head_size=12, num_heads=48, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.1, dropout=0.15):
                                                            head_size=12, num_heads=16, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.1, dropout=0.15):
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = tf.keras.layers.LayerNormalization()(inputs)
            x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = tf.keras.layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization()(res)
            x = tf.keras.layers.Conv2D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv2D(filters=inputs.shape[-1], kernel_size=1)(x)
            return x + res

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)
        trunk_outputs = tf.keras.layers.Dense(class_nb)(x)
        outputs = tf.keras.layers.Softmax()(trunk_outputs)

        return tf.keras.Model(inputs=inputs, outputs=trunk_outputs), tf.keras.Model(inputs=inputs, outputs=outputs)
            
    
    def train(self, train_set, checkpoint_path="model_saved", epochs=100, new_train=False):
        
        @tf.function
        def train_step(X, y):
            with tf.GradientTape() as tape: 
                pred = self.model(X, training=True)
                pred_loss = self.loss_fn(y, pred)
                total_loss = pred_loss 
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
#            tf.print("---Current Loss:", total_loss)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if not new_train:
            self.model.load_weights(checkpoint_path)
        else:
            for ep in tqdm(range(epochs), desc=f"Training {self.name}"):
                for X, y in train_set:
                    train_step(X, y)
                self.model.save_weights(checkpoint_path)
    
    def adv_loss_fn(self, X, t, rho):
            Z_logits = tf.reshape(self.trunk_model(X, training=False), (self.class_nb,))
            return tf.maximum(tf.maximum(tf.reduce_max(Z_logits[:t]), 
                                              tf.reduce_max(Z_logits[t+1:]))-Z_logits[t], rho)
            

    def dtwar_attack(self, X, t, path=None, alpha=0.1, beta=0.1, eta=1e-2, rho=-5, max_iter=1e3, delta_l2_loss=1, dtw_path_tightness=10):
        @tf.function
        def dtwar_gradient_step(X_adv, t, path, max_l2_loss, alpha, alpha_l2, rho):
            with tf.GradientTape() as tape:
                tape.watch(X_adv)
                dtw_dist = dtw_differntiable(path, X, X_adv)
                l2_dist = tf.minimum(tf.nn.l2_loss(X - X_adv), max_l2_loss)
                target_loss = self.adv_loss_fn(tf.reshape(X_adv, [1, 1, -1, self.channel_nb]), t, rho)
                dist_loss = tf.add(tf.multiply(alpha_l2, l2_dist), tf.multiply(alpha, dtw_dist))
                loss_ = tf.add(dist_loss, target_loss)
                G = tape.gradient(loss_, X_adv)
            return G, target_loss, dist_loss

        X = tf.reshape(X, (self.seg_size, self.channel_nb))
        min_X = 1.25 * np.min(X, axis=0)
        max_X = 1.25 * np.max(X, axis=0)
        max_l2_loss = tf.nn.l2_loss(X - (X+delta_l2_loss))
        noise = tf.convert_to_tensor(np.random.normal(0,0.01, X.shape))
        X_adv = tf.add(X, noise)
        CT_dtw_mat = np.arange(1,self.seg_size**2+1).reshape((self.seg_size, self.seg_size))
        if path is None:
            path = path_conversion(dtw_random_path(dtw_path_tightness, CT_dtw_mat), CT_dtw_mat)
        min_loss = np.inf
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float64)
        alpha_l2 = tf.convert_to_tensor(-beta, dtype=tf.float64)
        safe_Xadv = None
        for _ in tqdm(range(max_iter), desc="Attack", leave=False):
            G, target_loss, dist_loss = dtwar_gradient_step(X_adv, t, path, max_l2_loss, alpha, alpha_l2, rho)
            if (target_loss < 0) and (dist_loss < min_loss):
                safe_Xadv = tf.identity(X_adv)
                min_loss = dist_loss
            if (target_loss<=rho) and (dist_loss < min_loss):
                min_loss = dist_loss
                return tf.reshape(X_adv, (1, 1, self.seg_size, self.channel_nb))
            if np.any(np.isnan(G)):
                break
            X_adv = tf.clip_by_value(X_adv - eta * G, min_X, max_X)

        if self.adv_loss_fn(tf.reshape(X_adv, [1, 1, -1, self.channel_nb]), t, rho)>=0:
            if safe_Xadv is None:
                return tf.reshape(X_adv, (1, 1, self.seg_size, self.channel_nb))
            else:
                return tf.reshape(safe_Xadv, (1, 1, self.seg_size, self.channel_nb))
        else:
            return tf.reshape(X_adv, (1, 1, self.seg_size, self.channel_nb))

        
    def predict(self, X):
        return tf.argmax(self.model(X, training=False), 1)
        
    def predict_stmax(self, X):
        return self.trunk_model(X, training=False)
    
    def score(self, X, y):
        acc = tf.keras.metrics.Accuracy()
        acc.reset_states()
        pred = self.predict(X)
        acc.update_state(pred, y)
        return acc.result().numpy()
    
             
        
             