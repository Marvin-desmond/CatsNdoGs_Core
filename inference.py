import tensorflowjs as tfjs
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import glob

preprocessing = tf.keras.layers.experimental.preprocessing
Normalization = preprocessing.Normalization
Rescaling = preprocessing.Rescaling

class DefaultConfig:
    img_size: int = 224
    seed: int = 42
    gpu: int = tf.config.list_physical_devices('GPU')
    labels: list = ['cat', 'dog']


class ProductionModel(tf.keras.Model):
    def __init__(self, model,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(ProductionModel, self).__init__()
        variance = [i**2 for i in std]
        self.rescale_layer = Rescaling(1. / 255)
        self.norm_layer = Normalization(
            mean=mean,
            variance=variance
            )
        self.softmax_layer = tf.keras.layers.Softmax()
        self.model = model

    def call(self, x):
        x = self.rescale_layer(x)
        x = self.norm_layer(x)
        x = self.model(x)
        x = self.softmax_layer(x)
        return tf.argmax(tf.squeeze(x))

model = tf.keras.models.load_model("res-cats-dogs.h5")
prod_model = ProductionModel(model)
prod_model.save("prod-cats-dogs")

def read_image(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, 
        (DefaultConfig.img_size, 
        DefaultConfig.img_size)
        )
    return tf.expand_dims(image, axis=0)



image_paths = glob.glob("./imgs/run/*")
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    image = read_image(image_paths[i])
    pred = prod_model(image)
    pred = DefaultConfig.labels[pred]
    plt.imshow(image[0, ...])
    plt.title(pred)
    plt.axis("off")
plt.show()


# TFLITE
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(prod_model)

# converter.allow_custom_ops = True
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details,"\n",output_details)

plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(4, 4, i + 1)
    image = read_image(image_paths[i])
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred = DefaultConfig.labels[pred]
    plt.imshow(image[0, ...])
    plt.title(pred)
    plt.axis("off")
plt.show()


# TENSORFLOWJS
tfjs.converters.save_keras_model(prod_model, "./") # produces error
"""
=========================NOTE===============================
NotImplementedError:
Saving the model to HDF5 format requires the model to be a
Functional model or a Sequential model. It does not work for
subclassed models, because such models are defined via the body
of a Python method, which isn't safely serializable. Consider
saving to the Tensorflow SavedModel format
(by setting save_format="tf") or using `save_weights`.
"""

prod_model.save("cats_dogs_model")
