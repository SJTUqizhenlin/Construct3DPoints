import cv2
import numpy


shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink


def ExtractRnT(kp1, kp2, inlier_match1, mask2, E):
	matchesMask2 = mask2.ravel().tolist()
	inlier_match2 = []
	for i in range(len(inlier_match1)):
		if matchesMask2[i]:
			inlier_match2.append(inlier_match1[i])
	if len(inlier_match2) < 5:
		img_err = cv2.imread("error.png")
		cv2.imshow("Error", img_err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		exit()
	src_pts3 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match2 ]).reshape(-1, 1, 2)
	dst_pts3 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match2 ]).reshape(-1, 1, 2)
	M, R, t, mask3 = cv2.recoverPose(E, src_pts3, dst_pts3, K)
	return R, t, inlier_match2, mask3
