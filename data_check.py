import cv2
import random
from tqdm import tqdm
import glob
import os

files = []

def verify_jpeg_image(file_path):
    try:
        img = cv2.imread(file_path)
        shape = img.shape
    except:
        return False
    return True

removed = 0
corrupted = []
for f in tqdm(
    glob.glob('./kagglecatsanddogs_3367a/**/*.jpg', recursive=True)
    ):
    if (verify_jpeg_image(f)):
        files.append(f)
    else:
        corrupt = '_'.join(f.split("\\")[-2:])
        corrupted.append(f"Corrupt: {corrupt}")
        os.remove(f)
        removed += 1

a = True
for corrupt in tqdm(corrupted):
    if a:
        print(corrupt, end="")
        a = False
    else:
        print(f"\t{corrupt}")
        a = True

print(f"{removed} corrupted files removed")

random.shuffle(files)
print(len(files))
