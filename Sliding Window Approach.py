import cv2


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# Load an image
image = cv2.imread('images/zoo.png')
(window_width, window_height) = (200,200)  # Define window size

# Loop over the sliding window
for (x, y, window) in sliding_window(image, step_size=32, window_size=(window_width, window_height)):
    if window.shape[0] != window_height or window.shape[1] != window_width:
        continue
    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)

cv2.destroyAllWindows()
