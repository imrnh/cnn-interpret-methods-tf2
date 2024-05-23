import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import train_conf

def guided_backprop(model, image, last_cnn, classifier_layers):
    """
    Compute the guided backpropagation gradients for the given image and target class.
    """
    last_conv_layer = model.get_layer(last_cnn)
    last_conv_layer_model = keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layers:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = np.argmax(preds[0])
        pred_idx_value = preds[:, top_pred_index]

    grads = tape.gradient(pred_idx_value, last_conv_layer_output)
    guided_grads = tf.cast(grads > 0, "float32") * tf.cast(grads, "float32")
    guided_grads = tf.math.l2_normalize(guided_grads, axis=[0, 1, 2])

    return guided_grads

def guided_gradcam(model, image, last_cnn, classifier_layers):
    """
    Compute the Guided GradCAM heatmap for the given image.
    """
    last_conv_layer = model.get_layer(last_cnn)
    last_conv_layer_model = keras.Model(inputs=model.inputs, outputs=last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layers:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = np.argmax(preds[0])
        pred_idx_value = preds[:, top_pred_index]

    grads = tape.gradient(pred_idx_value, last_conv_layer_output)
    guided_grads = guided_backprop(model, image, last_cnn, classifier_layers)
    pooled_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.clip(heatmap, 0, np.max(heatmap)) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, train_conf.img_shape)

    return heatmap