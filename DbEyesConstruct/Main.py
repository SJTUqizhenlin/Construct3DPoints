import cv2
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sift
import calfundamental
import calessential
import calRandT
import triangulate
import draw3Dscatter

shrink = 0.2
K = numpy.zeros((3, 3), numpy.float32)
K[0][0] = 3401 * shrink
K[1][1] = 3401 * shrink
K[2][2] = 1
K[0][2] = 1488 * shrink
K[1][2] = 1984 * shrink

def main():
	name1 = "img7.jpg"
	name2 = "img8.jpg"
	
	img1, img2, kp1, kp2, nice_match = sift.Read_N_Sift(name1, name2)
	
	F, inlier0, mask1 = calfundamental.Calculate(kp1, kp2, nice_match)
	
	E, inlier1, mask2 = calessential.Calculate(kp1, kp2, inlier0, mask1)
	
	R, t, inlier2, mask3 = calRandT.ExtractRnT(kp1, kp2, inlier1, mask2, E)
	
	points4D, inlier3 = triangulate.Triang(kp1, kp2, inlier2, mask3, R, t)

	print("\n\nThe coordinates are listed below:\n\n", points4D, "\n\n")
	
	draw3Dscatter.Draw(kp1, img1, inlier3, points4D)


if __name__ == "__main__":
	main()
