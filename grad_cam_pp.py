import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import train_conf


def grad_cam_plus(model, img, layer_name, classifier_layers):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    last_conv_layer_model = keras.Model(inputs=model.inputs, outputs=conv_layer.output)

    print(conv_layer.output.shape[1:])
    classifier_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
    x = classifier_input
    
    for layer_name in classifier_layers:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output = last_conv_layer_model(img_tensor)
                predictions= classifier_model(conv_output)
                category_id = np.argmax(predictions[0])
                pred_idx_value = predictions[:, category_id]
            conv_first_grad = gtape3.gradient(pred_idx_value, conv_output)
        conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
    conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    a1 = conv_second_grad[0]
    a2 = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum

    alpha = a1/a2
    alpha_normalization_constant = np.sum(alpha, axis=(0,1))
    alpha /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alpha, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    heatmap = np.clip(heatmap, 0, np.max(heatmap)) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, train_conf.img_shape)

    return heatmap