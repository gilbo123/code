import numpy as np
import argparse
import imutils
import cv2
import os


#Image type
imagetypeRGB = 'RGBTop'
imagetypeIR = 'IRTop'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
	help="path to input image directory")
ap.add_argument("-p", "--output_dir_RGB", required=True,
	help="path to output passed image directory")
ap.add_argument("-f", "--output_dir_IR", required=True,
	help="path to output failed image directory")
args = vars(ap.parse_args())


IRCount = 0
RGBCount = 0
subdirs = [x[0] for x in os.walk(args["input_dir"])]                                                                            
for subdir in subdirs:                                                                                            
	files = os.walk(subdir).next()[2]
	if (len(files) > 0):
		for file in files:
			if (imagetypeRGB in file):
				print os.path.join(subdir, file)
				img = cv2.imread(os.path.join(subdir, file), -1) #-1 no change to image
				# small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
				# crop_img = small[0:600, 300:800] # Crop from x, y, w, h -> 100, 200, 800, 1200
				# cv2.imshow('image',crop_img)
				# k = cv2.waitKey(0)
				# if k == ord('p'): # wait for 's' key to save and exit
				# 	cv2.imwrite(os.path.join(args["output_dir_pass"], str(count) + ".jpg"),crop_img)
				# elif k == ord('f'): # wait for 'l' key to save and exit
				# 	cv2.imwrite(os.path.join(args["output_dir_fail"], str(count) + ".jpg") ,crop_img)
				# elif k == 27: # wait for esc key exit
				# 	cv2.destroyAllWindows()
				# 	break
				#make a "back button"
				#make a "Save to temp folder button"
				#make a "skip button"
				#add a count so I know when it ENDSSS!!!!
				cv2.imwrite(os.path.join(args["output_dir_RGB"], "RGB_" + str(RGBCount) + ".jpg") ,img)

				# if k == 27:
				# 	break #break out of the outer loop too

				RGBCount+=1
			elif (imagetypeIR in file):
				print os.path.join(subdir, file)
				img = cv2.imread(os.path.join(subdir, file), -1) #-1 no change to image
				cv2.imwrite(os.path.join(args["output_dir_IR"], "IR_" + str(IRCount) + ".jpg") ,img)

				IRCount+=1


# cv2.destroyAllWindows()


