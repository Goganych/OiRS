from shapedetector import ShapeDetector
import imutils
import cv2


# load the image and resize it to a smaller factor so that
# the shapes can be approximated better

image = cv2.imread("shapes.jpg")
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("gray.jpg", gray)

# распознавание контуров
# оператор Кэнни
# https://ru.wikipedia.org/wiki/%D0%9E%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80_%D0%9A%D1%8D%D0%BD%D0%BD%D0%B8
edged = cv2.Canny(gray, 10, 120)
cv2.imwrite("edged.jpg", edged)

#можно так сделать
# thresh = cv2.threshold(blurred, 10, 120, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (255, 192, 203), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_TRIPLEX,
		0.7, (255, 255, 255), 1)

	# show the output image
	cv2.imshow("Image", image)
	cv2.imwrite("output.jpg", image)
