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

def Draw(kp1, img1, inlier_match3, points4D):
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
