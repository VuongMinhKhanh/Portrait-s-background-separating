import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2 # Import the library OpenCV 

img = mpimg.imread('../pictures/girl3.png')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.title("Original portrait")
plt.show()

X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))


def RGBScale(image):
    if 0 < image[0][0][0] < 1:
        return (image * 255).astype(np.uint8)
    else:
        return image


def ClusterImage(image, nClusters):
    X = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    for K in [nClusters]:
        kmeans = KMeans(n_clusters=nClusters, n_init=10).fit(X)
        label = kmeans.predict(X)

        imgTemp = np.zeros_like(X)

        # replace each pixel by its center !!!
        for k in range(K):
            imgTemp[label == k] = kmeans.cluster_centers_[k]

        # reshape and display output image
        imgRes = imgTemp.reshape((image.shape[0], image.shape[1], image.shape[2]))

    return RGBScale(imgRes)


def PlotShowClusteredImages(clusteredImage, n, _interpolation, title):
    clusteredImage = RGBScale(clusteredImage)
    plt.clf()
    for K in [n]: 
        image = []
        if n != 0:
            image = ClusterImage(clusteredImage, K)
        else:
            image = clusteredImage.copy()
        plt.imshow(image, interpolation = _interpolation)
        plt.axis('off')
        plt.title(title)
        plt.show()
    return


def SegmentColorAreas(clusteredImage, pointList):
    for x in pointList:
        color_area = clusteredImage[x[0]][x[1]]

        # Create a mask for flood filling
        h, w = clusteredImage.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill the area
        cv2.floodFill(clusteredImage, mask, (x[1], x[0]), (255, 255, 255), loDiff=color_area.tolist(), upDiff=color_area.tolist())

    return clusteredImage


def SegmentAreas(originalImage, colorSegmentedImage):
    orginalImage = RGBScale(originalImage)

    # Find the coordinates of the filled area by comparing the original and flood-filled images
    indices = np.where(np.all(colorSegmentedImage == [255, 255, 255], axis=-1))

    # Change the pixels in the original image to transparent color based on the coordinates
    for y, x in zip(indices[0], indices[1]):
        originalImage[y, x] = [255, 255, 255]

    return originalImage
###

# Segment the picture into n colors / n clusters
n = 3
clusteredImg = ClusterImage(img, n)
PlotShowClusteredImages(clusteredImg, 3, 'bilinear', "Color clustered portrait")

# Segment selected color areas
pointList = [[20, 20],
             [220, 390]
    ]

colorSegmentedImg = SegmentColorAreas(clusteredImg, pointList)
#PlotShowClusteredImages(colorSegmentedImg, 0, 'nearest')

# Segment the original image
areaSegmentedImg = SegmentAreas(RGBScale(img), colorSegmentedImg) # cant RGBScale inside SegmentAreas
PlotShowClusteredImages(areaSegmentedImg, 0, 'bilinear', "Backgroundless portrait")



