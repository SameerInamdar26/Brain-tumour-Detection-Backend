import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image

GRAD_CAM_LAYER_NAME = "Conv_1"

def make_gradcam_heatmap(img_array, model, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(GRAD_CAM_LAYER_NAME).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, original_size, alpha=0.4):
    img = keras_image.img_to_array(original_img.resize(original_size))
    heatmap = np.uint8(255 * heatmap)

    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = keras_image.array_to_img(jet_colors[heatmap])
    jet_heatmap = jet_heatmap.resize(original_size)
    jet_heatmap = keras_image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras_image.array_to_img(superimposed_img / 255.0)

    return superimposed_img