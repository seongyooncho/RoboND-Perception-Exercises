import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('Udacican.png')
image *= 255

# Take histograms in R, G, and B
r_hist = np.histogram(image[:, :, 0], bins=32, range=(0, 256))
g_hist = np.histogram(image[:, :, 1], bins=32, range=(0, 256))
b_hist = np.histogram(image[:, :, 2], bins=32, range=(0, 256))

# Generating bin centers
bin_edges = r_hist[1]
bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

fig = plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.bar(bin_centers, r_hist[0])
plt.xlim(0, 256)
plt.title('R Histogram')
plt.subplot(132)
plt.bar(bin_centers, g_hist[0])
plt.xlim(0, 256)
plt.title('G Histogram')
plt.subplot(133)
plt.bar(bin_centers, b_hist[0])
plt.xlim(0, 256)
plt.title('B Histogram')
plt.show()

hist_feature = np.concatenate((r_hist[0], g_hist[0], b_hist[0])).astype(np.float64)

norm_feature = hist_feature / np.sum(hist_feature)

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range(0, 256)):
  # Convert from RGB to HSV using cv2.cvtColor()
  img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  # Compute the histogram of the HSV channels separately
  h_hist = np.histogram(img_hsv[:, :, 0], bins=32, range=(0, 256))
  s_hist = np.histogram(img_hsv[:, :, 1], bins=32, range=(0, 256))
  v_hist = np.histogram(img_hsv[:, :, 2], bins=32, range=(0, 256))
  # Concatenate the histograms into a single feature vector
  hist_feature = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
  # Normalize the result
  norm_feature = hist_feature / np.sum(hist_feature)
  return norm_feature
  
