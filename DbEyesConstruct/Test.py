import cv2
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink


def main():
	img1 = cv2.imread("img7.jpg")
	img2 = cv2.imread("img8.jpg")
	img1 = cv2.resize(img1, (0,0), fx=shrink, fy=shrink, 
		interpolation=cv2.INTER_CUBIC)
	img2 = cv2.resize(img2, (0,0), fx=shrink, fy=shrink, 
		interpolation=cv2.INTER_CUBIC)

	cv2.imshow("img1", img1)
	cv2.imshow("img2", img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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

	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, nice_match[:], None, flags=2)
	cv2.imshow("img3", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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

	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier_match0[:], None, flags=2)
	cv2.imshow("img3", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	src_pts1 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match0 ]).reshape(-1, 1, 2)
	dst_pts1 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match0 ]).reshape(-1, 1, 2)
	F, mask1 = cv2.findFundamentalMat(src_pts1, dst_pts1, cv2.FM_RANSAC, 1, 0.99)
	print("FundamentalMat is:\n", F)

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

	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier_match1[:], None, flags=2)
	cv2.imshow("img3", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	src_pts2 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match1 ]).reshape(-1, 1, 2)
	dst_pts2 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match1 ]).reshape(-1, 1, 2)
	E, mask2 = cv2.findEssentialMat(src_pts2, dst_pts2, K, cv2.RANSAC, 0.99, 1)
	print("EssentialMat is:\n", E)

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

	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier_match2[:], None, flags=2)
	cv2.imshow("img3", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	src_pts3 = numpy.float32([kp1[m[0].queryIdx].pt for m in inlier_match2 ]).reshape(-1, 1, 2)
	dst_pts3 = numpy.float32([kp2[m[0].trainIdx].pt for m in inlier_match2 ]).reshape(-1, 1, 2)
	M, R, t, mask3 = cv2.recoverPose(E, src_pts3, dst_pts3, K)
	print("Rotation is:\n", R)
	print("Translation is:\n", t)

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

	img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier_match3[:], None, flags=2)
	cv2.imshow("img3", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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
	print("Points are:\n", points4D)

	ptList = []
	x_3D = []
	y_3D = []
	z_3D = []
	for i in range(len(inlier_match3)):
		ptList.append(kp1[inlier_match3[i][0].queryIdx].pt)
		x_3D.append((points4D[0][i] / points4D[3][i]))
		y_3D.append((points4D[1][i] / points4D[3][i]))
		z_3D.append((points4D[2][i] / points4D[3][i]))
	blue = [img1[int(Pt[1]), int(Pt[0]), 0] for Pt in ptList]
	green = [img1[int(Pt[1]), int(Pt[0]), 1] for Pt in ptList]
	red = [img1[int(Pt[1]), int(Pt[0]), 2] for Pt in ptList]
	color_3D = []
	for i in range(len(ptList)):
		color_3D.append([red[i] / 255, green[i] / 255, blue[i] / 255])

	ax = plt.subplot(111, projection="3d")
	ax.scatter(x_3D, y_3D, z_3D, c=color_3D, marker=".")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	plt.show()


if __name__ == "__main__":
	main()