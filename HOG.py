import matplotlib.pyplot as plt
from skimage import feature, color, io

# Load an image
image = io.imread('images/doggo.jpg')
gray = color.rgb2gray(image)

# Compute HOG features and visualize
hog_features, hog_image = feature.hog(gray, visualize=True)

# Display the HOG image
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Features')
plt.axis(False)
plt.show()

plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis(False)
plt.show()
