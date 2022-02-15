import tensorflow as tf
import cv2
import albumentations as A
from albumentations import transforms
from tqdm import tqdm

AUTOTUNE = tf.data.AUTOTUNE

Normalize = transforms.Normalize
CoarseDropout = transforms.CoarseDropout
Cutout = transforms.Cutout
Flip = transforms.Flip

class createDataset():
    def __init__(self, paths, batch_size, img_size, training):
        super(createDataset, self).__init__()
        self.paths = paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.training = training
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
        self.train_transforms = A.Compose([
                    Normalize(mean=mean, std=std, always_apply=True, p=1.0),
                    CoarseDropout(max_holes=10, max_height=5, max_width=5, fill_value=64),
                    Flip(),
        ])

        self.test_transforms = A.Compose([
                    Normalize(mean=mean, std=std, always_apply=True, p=1.0),
        ])

    def read_inputs(self, path):
        path = (path.numpy()).decode("utf-8")
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, [self.img_size, self.img_size])
        if self.training:
            label = (path.split("\\")[-2])
            label = 0 if label == 'Cat' else 1
            label = tf.keras.utils.to_categorical(label, num_classes=2)
            return (image, label)
        else:
            return image

    def augment_image(self, image, training=False):
      image = image.numpy()
      if training:
        image = self.train_transforms(image=image)["image"]
      else:
        image = self.test_transforms(image=image)["image"]
      return image

    def preprocess_data(self, image, label=None, training=False):
      image = tf.py_function(func=self.augment_image, inp=[image, training], Tout=tf.float32)
      if training:
        return image, label
      else:
        return image

    def load_data(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)
        if self.training:
            dataset = dataset.map(
                lambda x: tf.py_function(
                    func=self.read_inputs,
                    inp=[x],
                    Tout=[tf.float32, tf.float32]
                    ),
                    num_parallel_calls= AUTOTUNE
                ).map(
                lambda x, y : self.preprocess_data(x, y, self.training),
                num_parallel_calls=AUTOTUNE
            ).shuffle(16).batch(
            self.batch_size
            ).prefetch(
            tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                lambda x: tf.py_function(
                    func=self.read_inputs,
                    inp=[x],
                    Tout=[tf.float32]
                    ),
                    num_parallel_calls= AUTOTUNE
                ).map(
                lambda x : self.preprocess_data(x, None),
                num_parallel_calls=AUTOTUNE
            ).shuffle(16).batch(
            self.batch_size
            ).prefetch(
            tf.data.AUTOTUNE
            )
        return dataset
