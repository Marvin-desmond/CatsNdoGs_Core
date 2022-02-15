import tensorflow as tf
import cv2
import albumentations as A
from albumentations import transforms
from tqdm import tqdm

Normalize = transforms.Normalize
CoarseDropout = transforms.CoarseDropout
Cutout = transforms.Cutout
MultiplicativeNoise = transforms.MultiplicativeNoise
Flip = transforms.Flip

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transforms = A.Compose([
            Normalize(mean=mean, std=std, always_apply=True, p=1.0),
            CoarseDropout(max_holes=30, max_height=10, max_width=10, fill_value=64),
            MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True),
            Flip(),
])

test_transforms = A.Compose([Normalize(mean=mean, std=std, always_apply=True, p=1.0),])

def read_inputs(path, size, training):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, [size, size])
    if training:
        label = (path.split("\\")[-2])
        label = 0 if label == 'Cat' else 1
        label = tf.keras.utils.to_categorical(label, num_classes=2)
        return (image, label)
    else:
        return image

def create_dataset(image_paths, batch_size, image_size, training):
    # creating generator
    def generator():
        for path in image_paths:
            if training:
                try:
                    image, label = read_inputs(path, image_size, training)
                    image = train_transforms(image=image)["image"]
                    yield image, label
                except:
                    print(f"Corrupted: {path}")
            else:
                try:
                    image = read_inputs(path, image_size, training)
                    image = test_transforms(image=image)["image"]
                    yield image
                except:
                    print(f"Corrupted: {path}")
    # Loading with generator to the computational graph
    if training:
        data = tf.data.Dataset.from_generator(
            generator,
              output_signature=(
                      tf.TensorSpec(
                          shape=(
                          image_size,
                          image_size,
                          3),
                      dtype=tf.float32),
                      tf.TensorSpec(
                          shape=(2,),
                          dtype=tf.int32)
              )
            )
    else:
        data = tf.data.Dataset.from_generator(
            generator,
              output_signature=(
                      tf.TensorSpec(
                          shape=(
                          image_size,
                          image_size,
                          3),
                      dtype=tf.float32),
              )
            )

    return data.batch(batch_size)

# from data_loader_gen import create_dataset
# dataset = create_dataset(
#             test_paths,
#             DefaultConfig.batch_size,
#             DefaultConfig.img_size,
#             training=True)
