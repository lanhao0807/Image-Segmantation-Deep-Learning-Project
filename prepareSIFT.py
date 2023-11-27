import scipy.io
import matplotlib.pyplot as plt
import os
import cv2


# Load .mat file
# dict_keys(['__header__', '__version__', '__globals__', 'Images', 'Label'])
mat_data = scipy.io.loadmat('SIFT/sift_flow.mat')
print(mat_data['Images'].shape)
print(mat_data['Label'].shape)

images = mat_data['Images']
labels = mat_data['Label']

index = 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(images[index])
axes[0].set_title("Image")
axes[0].axis('off')

axes[1].imshow(labels[index])
axes[1].set_title("Label")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Create the directories for saving the images and labels
# if not os.path.exists("SIFT/images"):
#     os.makedirs("SIFT/images")
# if not os.path.exists("SIFT/labels"):
#     os.makedirs("SIFT/labels")

# Iterate over the images and labels, and save them individually
# for i in range(images.shape[0]):
#     # Save the image
#     img_path = os.path.join("SIFT/images", f"{i}.jpg")
#     cv2.imwrite(img_path, images[i])

#     # Save the label
#     label_path = os.path.join("SIFT/labels", f"{i}.jpg")
#     cv2.imwrite(label_path, labels[i])



