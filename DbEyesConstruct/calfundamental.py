import cv2
import numpy


shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink


def Calculate(kp1, kp2, nice_match):
	src_pts0 = numpy.float32([kp1[m[0].queryIdx].pt for m in nice_match ]).reshape(-1, 1, 2)
	dst_pts0 = numpy.float32([kp2[m[0].trainIdx].pt for m in nice_match ]).reshape(-1, 1, 2)
	H, mask0 = cv2.findHomography(src_pts0, dst_pts0, cv2.RANSAC, 3, None, 100, 0.99)
	matchesMask0 = mask0.ravel().tolist()
	inlier_match0 = []
	for i in range(len(nice_match)):
		if matchesMask0[i]:
			inlier_match0.append(nice_match[i])
	if len(inlier_match0) < 8:
		img_err = cv2.imread("error.png")
		cv2.imshow("Error", img_err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		exit()
	src_pts1 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match0 ]).reshape(-1, 1, 2)
	dst_pts1 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match0 ]).reshape(-1, 1, 2)
	F, mask1 = cv2.findFundamentalMat(src_pts1, dst_pts1, cv2.FM_RANSAC, 1, 0.99)
	return F, inlier_match0, mask1
