import cv2
import numpy


shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink


def Read_N_Sift(imgname1, imgname2):
	img1 = cv2.imread(imgname1)
	img2 = cv2.imread(imgname2)
	img1 = cv2.resize(img1, (0,0), fx=shrink, fy=shrink, 
		interpolation=cv2.INTER_CUBIC)
	img2 = cv2.resize(img2, (0,0), fx=shrink, fy=shrink, 
		interpolation=cv2.INTER_CUBIC)
	sift = cv2.xfeatures2d.SIFT_create(2000)
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	if len(matches) < 10:
		img_err = cv2.imread("error.png")
		cv2.imshow("Error", img_err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		exit()
	nice_match = []
	for m, n in matches:
		if m.distance < 0.8 * n.distance:
			nice_match.append([m])
	if len(nice_match) < 10:
		img_err = cv2.imread("error.png")
		cv2.imshow("Error", img_err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		exit()
	return img1, img2, kp1, kp2, nice_match
