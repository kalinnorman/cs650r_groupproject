from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

'''THIS FILE ASSUMES THAT IT WILL BE RUN FROM THE ROOT DIRECTORY OF THE REPO'''

# Open the images
pim1 = Image.open('test_imgs/img_0000.dng')
pim2 = Image.open('test_imgs/img_0001.dng')
pim3 = Image.open('test_imgs/img_0002.dng')

# Get the pixel data
px1 = list(pim1.getdata())
px2 = list(pim2.getdata())
px3 = list(pim3.getdata())

# Reshape pixel data into 3D array
im1 = np.array(px1).reshape((pim1.size[1], pim1.size[0], 3))
im2 = np.array(px2).reshape((pim2.size[1], pim2.size[0], 3))
im3 = np.array(px3).reshape((pim3.size[1], pim3.size[0], 3))

# Set up plots
f1, a1 = plt.subplots()
f2, a2 = plt.subplots()
f3, a3 = plt.subplots()

# Display images on plots
a1.imshow(im1)
a2.imshow(im2)
a3.imshow(im3)

# Show the plots
plt.show()

'''
List of image correspondences:
(These were identified by zooming into each photo and recording the pixel locations of features that I found)

| im1 [x, y]  | im2 [x, y]  | im3 [x, y]  | Description                                                                 |
| ----------- | ----------- | ----------- | --------------------------------------------------------------------------- |
| [455, 792]  | [243, 740]  | [253, 452]  | Center of windows key                                                       |
| [667, 513]  | [598, 574]  | [553, 465]  | Bottom right of h in hp logo on keyboard                                    |
| [1194, 467] | [1268, 797] | [930, 939]  | First non white pixel center above INPHIC logo on mouse                     |
| [692, 348]  | [781, 412]  | [824, 422]  | Bottom right corner of wedding photo                                        |
| [295, 694]  | [203, 600]  | [302, 356]  | Brightest pixel of grave symbol (top left of keyboard, also the tilde key)  |
| [986, 484]  | [963, 692]  | [745, 698]  | Top left of hole in 9 on numpad                                             |
| [482, 76]   | [608, 86]   | [776, 64]   | Top left corner of wedding photo                                            |
| [922, 47]   | [1155, 137] | [1249, 262] | Bottom right of the h in the hp logo on the monitor                         |

'''

