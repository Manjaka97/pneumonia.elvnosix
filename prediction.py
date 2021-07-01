import numpy as np
import cv2


def extract_boxes_confidences_classids(outputs, confidence, width, height):
	boxes = []
	confidences = []
	classIDs = []

	for output in outputs:
		for detection in output:
			# Extract the scores, classid, and the confidence of the prediction
			scores = detection[5:]
			classID = np.argmax(scores)
			conf = scores[classID]

			# Consider only the predictions that are above the confidence threshold
			if conf > confidence:
				# Scale the bounding box back to the size of the image
				box = detection[0:4] * np.array([width, height, width, height])
				centerX, centerY, w, h = box.astype('int')

				# Use the center coordinates, width and height to get the coordinates of the top left corner
				x = int(centerX - (w / 2))
				y = int(centerY - (h / 2))

				boxes.append([x, y, int(w), int(h)])
				confidences.append(float(conf))
				classIDs.append(classID)

	return boxes, confidences, classIDs


def draw_bounding_boxes(image, labels, boxes, confidences, classIDs, idxs, color):
	if len(idxs) > 0:
		for i in idxs.flatten():
			# extract bounding box coordinates
			x, y = boxes[i][0], boxes[i][1]
			w, h = boxes[i][2], boxes[i][3]

			# draw the bounding box and label on the image
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			# text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i]*100)
			lb = "{}".format(labels[classIDs[i]])
			conf = "{:.2f}%".format(confidences[i] * 100)
			cv2.putText(image, lb, (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			cv2.putText(image, conf, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

	return image


def make_prediction(net, layer_names, labels, image, confidence, threshold):
	height, width = image.shape[:2]

	# Create a blob and pass it through the model
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(layer_names)

	# Extract bounding boxes, confidences and classIDs
	boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

	# Apply Non-Max Suppression
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

	return boxes, confidences, classIDs, idxs


def detect(img_path, labels=['Possible Pneumonia Sign'], colors=(0, 0, 255), config='yolov4.cfg',
		   weights='yolov4.weights', conf=.001, thresh=.001):
	# Load weights using OpenCV
	net = cv2.dnn.readNetFromDarknet(config, weights)

	# Get the ouput layer names
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	image = cv2.imread(img_path)

	boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, conf, thresh)

	image = draw_bounding_boxes(image, labels, boxes, confidences, classIDs, idxs, colors)

	cv2.imwrite(img_path, image)
	cv2.destroyAllWindows()

