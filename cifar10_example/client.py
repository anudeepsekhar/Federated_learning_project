#%%
import flwr as fl
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
from tensorflow import keras
import tensorflow_datasets as tfds
import sys
import os
#%%
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  X = tf.image.resize(image,[224,224])
  X = tf.keras.applications.efficientnet.preprocess_input(X)
  return X, label

#%%
# Load and compile Keras model
n_classes = 10
base_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3),
)
base_model.trainable = False
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
#%%
# Load dataset
splits = {
    '1':'[:25%]',
    '2':'[25%:50%]',
    '3':'[50%:75%]',
    '4':'[75%:]'
}

n_client = str(sys.argv[2])
print(f'Selected Split: {n_client}')
batch_size = 32


(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train'+splits[n_client], 'test'+splits[n_client]],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)


ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
#%%
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(ds_train, epochs=5)
        return model.get_weights(), len(ds_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(ds_test)
        return loss, len(ds_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
