import numpy as np
import argparse
import cv2
import os
import sys
import datetime


#Image type
imagetype = 'IRTop'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
	help="path to input image directory")
ap.add_argument("-p", "--output_dir", required=True,
	help="path to output passed image directory")
args = vars(ap.parse_args())




def save_index():
	global total, passCount, underCount, overCount, smallCount, foreignCount, bruiseCount, interestCount
	#save the input dir index to txt file
	with open('index.txt','w') as f:
		f.write(str(total) +'\n')
		f.write(str(passCount) +'\n')
		f.write(str(underCount) +'\n')
		f.write(str(overCount) +'\n')
		f.write(str(smallCount) +'\n')
		f.write(str(foreignCount) +'\n')
		f.write(str(interestCount))

def load_index():
	#load the index from txt file
	global total, passCount, underCount, overCount, smallCount, foreignCount, bruiseCount, interestCount
	content = []
	if os.path.isfile('index.txt'):
		with open('index.txt','r') as f:
			#read file
			content = [x.strip('\n') for x in f.readlines()]
			# load contents
			total = int(content[0])
			passCount = int(content[1])
			underCount = int(content[2])
			overCount = int(content[3])
			smallCount = int(content[4])
			foreignCount = int(content[5])
			interestCount = int(content[6])

	print("Starting at punnet no: {}".format(total))


'''
MAIN
'''
def main():
	winname = "Image"
	#some globals
	global total, passCount, underCount, overCount, smallCount, foreignCount, bruiseCount, interestCount, subdirs
	global pass_folder, under_folder, over_folder, small_folder, foreign_folder, interesting_folder, userName
	for i in range(total, len(subdirs)):
		files = os.walk(subdirs[i]).next()[2]
		if (len(files) > 0):
			for file in files:
				if (imagetype in file):
					print os.path.join(subdirs[i], file)
					img = cv2.imread(os.path.join(subdirs[i], file), -1) #-1 no change to image
					small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
					if imagetype == 'RGBTop':
						crop_img = small[0:600, 300:800] # Crop row1, row2, col1, col2
					else:
						crop_img = small[0:600, 200:700]
					cv2.namedWindow(winname)        # Create a named window
					cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
					cv2.imshow(winname, crop_img)
					#instructions
					print("Please choose:")
					print("1 - underripe")
					print("2 - overripe")
					print("3 - too small")
					print("4 - foreign object")
					print("5 - bruise")
					print("p - pass")
					print("s - save intersting image")
					print("any key - drop (skip)")
					print("esc - save progress and stop\n")
					k = cv2.waitKey(0)

					#Wait for intruction
					#PASS
					if k == ord('p'):
						if not os.path.exists(pass_folder):
							os.makedirs(pass_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(passCount))
						cv2.imwrite(os.path.join(pass_folder, name + ".jpg"),crop_img)
						passCount+=1
					#UNDERRIPE
					elif k == ord('1'):
						if not os.path.exists(under_folder):
							os.makedirs(under_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(passCount))
						cv2.imwrite(os.path.join(under_folder, name + ".jpg") ,crop_img)
						underCount+=1
					#OVERRIPE
					elif k == ord('2'):
						if not os.path.exists(over_folder):
							os.makedirs(over_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(passCount))
						cv2.imwrite(os.path.join(over_folder, name + ".jpg") ,crop_img)
						overCount+=1
					#SMALL
					elif k == ord('3'):
						if not os.path.exists(small_folder):
							os.makedirs(small_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(passCount))
						cv2.imwrite(os.path.join(small_folder, name + ".jpg") ,crop_img)
						smallCount+=1
					#FOREIGN
					elif k == ord('4'):
						if not os.path.exists(foreign_folder):
							os.makedirs(foreign_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(passCount))
						cv2.imwrite(os.path.join(foreign_folder, name + ".jpg") ,crop_img)
						foreignCount+=1
					#BRUISE
					elif k == ord('5'):
						if not os.path.exists(bruise_folder):
							os.makedirs(bruise_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(bruiseCount))
						cv2.imwrite(os.path.join(bruise_folder, name + ".jpg") ,crop_img)
						bruiseCount+=1
					#SAVE INTERESTING IMAGE
					elif k == ord('s'):
						if not os.path.exists(interesting_folder):
							os.makedirs(interesting_folder)
						date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
						name = '{}_{}_{}'.format(userName, date, str(passCount))
						cv2.imwrite(os.path.join(interesting_folder, name + ".jpg") ,crop_img)
						interestCount+=1
					#QUIT
					# wait for esc key exit
					elif k == 1048603 or k == 27: #depends on OpenCV version???
						cv2.destroyAllWindows()
						save_index()
						print("exiting")
						exit()
					#make a "back button"

					# print("Button: {}".format(k))


					#increment total (index)
					total+=1
					cv2.destroyAllWindows()


if __name__ == '__main__':
	subdirs = [x[0] for x in os.walk(args["input_dir"])]
	print("number of directories: {}".format(len(subdirs)))
	#folders for classes
	pass_folder = args["output_dir"] + 'pass'
	under_folder = args["output_dir"] + 'under'
	over_folder = args["output_dir"] + 'over'
	small_folder = args["output_dir"] + 'small'
	foreign_folder = args["output_dir"] + 'foreign'
	bruise_folder = args["output_dir"] + 'bruise'
	interesting_folder = args["output_dir"] + 'interesting'

	#get a unique name for truthing owner
	cwd = os.getcwd()
	userName = cwd.split('/')[2]

	#start from zero
	passCount = underCount = overCount = smallCount = foreignCount = bruiseCount = interestCount = total = 0
	load_index()
	main()
