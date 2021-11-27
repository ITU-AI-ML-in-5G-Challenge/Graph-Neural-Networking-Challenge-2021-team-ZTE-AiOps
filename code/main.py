import tensorflow as tf
import sys

sys.path.insert(1, "./code")
from read_dataset import input_fn
from routenet_model import RouteNetModel
import configparser

import os

# In case you want to disable GPU execution uncomment this line
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def transformation(x, y):
    """Apply a transformation over all the samples included in the dataset.
           Args:
               x (dict): predictor variables.
               y (array): target variable.
           Returns:
               x,y: The modified predictor/target variables.
    """
    return x, y


# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

# Initialize the datasets
ds_train = input_fn(config['DIRECTORIES']['train'], shuffle=True)
ds_train = ds_train.map(lambda x, y: transformation(x, y))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_test = input_fn(config['DIRECTORIES']['test'], shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Define an exponential decay schedule.
decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(config['HYPERPARAMETERS']['learning_rate']),
                                                                int(config['HYPERPARAMETERS']['decay_steps']),
                                                                float(config['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=True)
# Define the optimizer
#optimizer = tf.keras.optimizers.Adam(learning_rate=float(config['HYPERPARAMETERS']['learning_rate']))
optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)

# Define, build and compile the model
model = RouteNetModel(config)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics="MAPE")

# Define the checkpoint directory where the model will be saved
ckpt_dir = config['DIRECTORIES']['logs']
latest = tf.train.latest_checkpoint(ckpt_dir)

# Reload the pretrained model in case it exists
if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}-{val_MAPE:.2f}")

# If save_best_only, the program will only save the best model using 'monitor' as metric
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode='min',
    monitor='val_MAPE',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

# This method trains the model saving the model each epoch.
trainMode = config['RUN_CONFIG']['train']

if trainMode == 'True' or trainMode == 'true':
    print("Start per-link Occupancy training...")
    model.fit(ds_train,
              epochs=int(config['RUN_CONFIG']['epochs']),
              steps_per_epoch=int(config['RUN_CONFIG']['steps_per_epoch']),
              validation_data=ds_test,
              validation_steps=int(config['RUN_CONFIG']['validation_steps']),
              callbacks=[cp_callback],
              use_multiprocessing=True)

from routenet_model_predict import RouteNetModelPredict
from read_dataset_predict import input_fn_predict

ds_predict = input_fn_predict(config['DIRECTORIES']['test'], shuffle=False)
ds_predict = ds_predict.map(lambda x, y: transformation(x, y))
ds_predict = ds_predict.prefetch(tf.data.experimental.AUTOTUNE)

# This method evaluates the trained model and outputs the desired metrics for all the test dataset.
#model.evaluate(ds_test)

model_predict = RouteNetModelPredict(config)
model_predict.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics="MAPE")
latest = tf.train.latest_checkpoint(ckpt_dir)
model_predict.load_weights(latest)

# train on delay target
trainMode1 = config['RUN_CONFIG']['train1']
if trainMode1 == 'True' or trainMode1 == 'true':
    print("Start per-Path delay training...")
    cp_callback1 = tf.keras.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    verbose=1,
                    mode='min',
                    monitor='val_MAPE',
                    save_best_only=False,
                    save_weights_only=True,
                    save_freq='epoch')
    ds_train1 = input_fn_predict(config['DIRECTORIES']['train'], shuffle=False)
    ds_train1 = ds_train1.map(lambda x, y: transformation(x, y))
    ds_train1 = ds_train1.prefetch(tf.data.experimental.AUTOTUNE)
    model_predict.fit(ds_train1,
              epochs=int(config['RUN_CONFIG']['epochs']),
              steps_per_epoch=int(config['RUN_CONFIG']['steps_per_epoch']),
              validation_data=ds_predict,
              validation_steps=int(config['RUN_CONFIG']['validation_steps']),
              callbacks=[cp_callback1],
              use_multiprocessing=True)

print("Start per-Path delay evaluate...")
model_predict.evaluate(ds_predict)

# This method return the predictions in a python array
#predictions = model.predict(ds_test)
#predictions = model_predict.predict(ds_predict)

# Do stuff here
#print(predictions)