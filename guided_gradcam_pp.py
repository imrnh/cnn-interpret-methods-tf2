import numpy as np
import tensorflow as tf
from tensorflow import keras

def guided_backprop(model, img, layer_name, category_id=None):
    """
    Compute the guided backpropagation gradients for the given image and target class.
    """
    conv_layer = model.get_layer(layer_name)
    heatmap_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                img_tensor = np.expand_dims(img, axis=0)
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)

            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    guided_grads = tf.cast(conv_first_grad > 0, "float32") * tf.cast(conv_first_grad, "float32")
    guided_grads = tf.math.l2_normalize(guided_grads, axis=[0, 1, 2])

    return guided_grads

def guided_grad_cam_plus_plus(model, img, layer_name, category_id=None):
    """
    Compute the Guided GradCAM++ heatmap for the given image.
    """
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)
    heatmap_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
                conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    guided_grads = guided_backprop(model, img, layer_name, category_id)
    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    a1 = conv_second_grad[0]
    a2 = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    a2 = np.where(a2 != 0.0, a2, 1e-10)
    alpha = a1 / a2
    alpha_normalization_constant = np.sum(alpha, axis=(0, 1))
    alpha /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)
    deep_linearization_weights = np.sum(weights * alpha, axis=(0, 1))
    grad_cam_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

    guided_grad_cam_map = guided_grads[0] * grad_cam_map
    heatmap = np.maximum(guided_grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap