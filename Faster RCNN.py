import cv2
import numpy as np

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
              "eye glasses", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
              "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
              "oven", "toaster", "sink", "refrigerator", "blender", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

weightsPath = "materials/frozen_inference_graph.pb"
configPath = "materials/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


def detect(image_path):
    # Load the image
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

    # Set the input to the pre-trained deep learning model
    net.setInput(blob)

    # Perform the forward pass to get the bounding boxes
    boxes = net.forward(["detection_out_final"])
    output = boxes[0].squeeze()

    # Filter out weak detections by ensuring the confidence is greater than a threshold
    confidence_threshold = 0.8
    num = np.argwhere(output[:, 2] > confidence_threshold).shape[0]

    # Load the image again for drawing bounding boxes
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(num):
        # Extract the bounding box coordinates
        x1n, y1n, x2n, y2n = output[i, 3:]
        x1 = int(x1n * W)
        y1 = int(y1n * H)
        x2 = int(x2n * W)
        y2 = int(y2n * H)

        # Draw the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Get the class name for the detected object
        class_name = coco_names[int(output[i, 1])]

        # Put the class name text above the bounding box
        img = cv2.putText(img, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the result
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect("images/traffic.jpg")
