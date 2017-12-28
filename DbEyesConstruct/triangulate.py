import cv2
import numpy


shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink


def Triang(kp1, kp2, inlier_match2, mask3, R, t):
	matchesMask3 = mask3.ravel().tolist()
	inlier_match3 = []
	for i in range(len(inlier_match2)):
		if matchesMask3[i]:
			inlier_match3.append(inlier_match2[i])
	if len(inlier_match3) < 5:
		img_err = cv2.imread("error.png")
		cv2.imshow("Error", img_err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		exit()
	src_pts4 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match3 ]).reshape(-1, 1, 2)
	dst_pts4 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match3 ]).reshape(-1, 1, 2)
	src_pts4 = (src_pts4.reshape(-1, 2)).transpose(1,0)
	dst_pts4 = (dst_pts4.reshape(-1, 2)).transpose(1,0)
	proj1 = numpy.column_stack((numpy.eye(3, dtype=numpy.float32), 
		numpy.zeros((3, 1), dtype=numpy.float32)))
	proj2 = numpy.column_stack((R, t))
	proj1_ = numpy.dot(K, proj1)
	proj2_ = numpy.dot(K, proj2)
	points4D = cv2.triangulatePoints(proj1, proj2, src_pts4, dst_pts4)
	return points4D, inlier_match3
