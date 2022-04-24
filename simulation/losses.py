#%%
import tensorflow as tf
import flwr as fl
import os
import dataset
from tqdm import tqdm
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# %%
partitions = dataset.load(num_partitions=2)
#%%
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.summary()
# global_model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# global_model.trainable = False
#%%
(x_train, y_train), (x_test, y_test) = partitions[0]
# %%
def custom_loss(W,Wt,mu):
    def calc_prox_np(w,wt):
        prox = np.linalg.norm(w-wt)
        print(prox)
        return prox
    def _custom_loss():
        prox = [tf.numpy_function(calc_prox_np, [w,wt], tf.float32) for w,wt in zip(W,Wt)]
        return tf.reduce_mean(prox)*mu
    return _custom_loss

#%%
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
global_model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
global_model.trainable = False
global_model.set_weights(model.weights)
Wt_init = global_model.weights
W_init = model.weights
prox = tf.reduce_mean([tf.norm(w-wt) for w,wt in zip(W_init,Wt_init)])
print(prox)
# inputs = inputs = tf.keras.layers.Input(shape=(32, 32, 3))
# outputs = sub_model(inputs)
# model = tf.keras.Model(inputs, outputs)
#%%
Wt = global_model.weights
W = model.weights
model.add_loss(custom_loss(W,Wt,0.1))
#%%
model.losses
# %%
model.compile("adam", ["sparse_categorical_crossentropy"], metrics=["accuracy"])
#%%
model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=100)
# %%
W = model.get_weights()
prox = tf.reduce_mean([tf.norm(w-wt) for w,wt in zip(W,Wt_init)])
print(prox)
# %%