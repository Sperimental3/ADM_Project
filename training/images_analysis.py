# code for image analysis, to understand the dimensions and how to pre-process the images
# before feeding them to the network

import os
import numpy as np
import cv2
# import torch
# import matplotlib.pyplot as plt


inPath = "C:\\Users\\Matteo\\Desktop\\Automated Decision Making\\Progetto\\dogs-vs-cats\\train\\train"

dataset_length = len(os.listdir(inPath))
print(dataset_length)

dataset_list = os.listdir(inPath)
dataset_list.sort(key=lambda x: (x.split(".")[0], int(x.split(".")[1])))

# print(dataset_list)

# array for analyzing images dimensions
dimensions = np.empty((len(os.listdir(inPath)), 3))

for imagePath, i in zip(dataset_list, range(dataset_length)):
    try:
        ex = cv2.imread(f"{inPath}\\{imagePath}")
        dimensions[i] = ex.shape
    except Exception as inst:
        print(type(inst))
        print("Some problem occurs.")

# print(dimensions)
print(dimensions.max(axis=0), dimensions.min(axis=0), dimensions.mean(axis=0))

"""
img = cv2.imread(f"{inPath}\\{os.listdir(inPath)[0]}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(type(img), img.shape)

plt.imshow(img)
plt.show()

img = torch.from_numpy(np.moveaxis(img, 2, 0)).unsqueeze_(0).type(torch.float32)

print(img.type(), img.shape)

output = torch.moveaxis(torch.squeeze(img), 0, 2).type(torch.int32)

print(output.type(), output.shape)
# print(output)

plt.imshow(output)
plt.show()
"""
