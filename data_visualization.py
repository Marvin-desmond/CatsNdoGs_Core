import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

class Visualization():
    def __init__(self):
        super(Visualization, self).__init__()
        self.labels: list = ['cat', 'dog']

    def inverse_norm_func(self, image, **kwargs):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image = (
            (np.asarray(image) * np.array(std) * 255) +
            (np.array(mean) * 255)).astype(np.uint8)
        return image

    def visualize(self, batch, training=True):
        inverse_norm = A.Lambda(
            name='inverse_norm', 
            image=self.inverse_norm_func, 
            p=1.0)
        if training:
            images, labels = batch
        else:
            images, labels = batch, np.ones((16,))
        plt.figure(figsize=(10, 10))
        for index, (image, label) in enumerate(zip(images, labels)):
            plt.subplot(4, 4, index + 1)
            if training:
                plt.title(self.labels[np.argmax(label)])
            plt.axis("off")
            image = inverse_norm(image=image)['image']
            plt.imshow(image)
        plt.show()
