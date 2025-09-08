# convert_to_tflite.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_H5 = "mobilenet_biogas.h5"   # produced earlier
TFLITE_OUT = "model_int8.tflite"
IMG_SIZE = 224
REP_SAMPLES = 200

# load keras model
model = load_model(MODEL_H5)

# Representative dataset generator (yield float32 arrays)
def representative_dataset_gen():
    # expects dataset directory with images for representative samples
    rep_dir = "dataset/rep"  # put ~200 varied images here (actual images from your camera)
    files = [os.path.join(rep_dir, f) for f in os.listdir(rep_dir) if f.lower().endswith((".jpg",".png"))]
    files = files[:REP_SAMPLES]
    for f in files:
        img = tf.keras.preprocessing.image.load_img(f, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = tf.expand_dims(arr, 0)
        arr = tf.cast(arr, tf.float32)
        # apply same preprocess as during training
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        yield [arr]

# Convert with int8 full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# set input/output types to uint8 or int8 (use uint8 for many Coral /Pi pipelines)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite model to", TFLITE_OUT)
