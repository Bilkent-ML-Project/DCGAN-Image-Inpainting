import os
import matplotlib.pyplot as plt
from skimage.transform import resize

# root path depends on your computer
root = "archive/img_align_celeba/img_align_celeba/"
save_root = "archive/img_align_celeba_resized/"
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root):
    os.mkdir(save_root)
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = resize(img, (resize_size, resize_size))
    fname = save_root + img_list[i][:-3] + "png"
    plt.imsave(fname=fname, arr=img, format="png")

    if (i % 1000) == 0:
        print("%d images complete" % i)
