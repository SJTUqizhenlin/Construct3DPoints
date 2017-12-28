import cv2
import numpy


shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink


def Calculate(kp1, kp2, inlier_match0, mask1):
	matchesMask1 = mask1.ravel().tolist()
	inlier_match1 = []
	for i in range(len(inlier_match0)):
		if matchesMask1[i]:
			inlier_match1.append(inlier_match0[i])
	if len(inlier_match1) < 5:
		img_err = cv2.imread("error.png")
		cv2.imshow("Error", img_err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		exit()
	src_pts2 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match1 ]).reshape(-1, 1, 2)
	dst_pts2 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match1 ]).reshape(-1, 1, 2)
	E, mask2 = cv2.findEssentialMat(src_pts2, dst_pts2, K, cv2.RANSAC, 0.99, 1)
	return E, inlier_match1, mask2
