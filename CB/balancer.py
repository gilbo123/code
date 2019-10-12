import sys
import os
import numpy as np

#source folder of labels
label_src = 'Labels/'

#list o' files
files = os.listdir(label_src)


#increment size based on image size
IMAGE_X = 1225
IMAGE_Y = 600
X_INCREMENT = 1/IMAGE_X
Y_INCREMENT = 1/IMAGE_Y



'''
EXTRACT class totals
'''
def get_class_totals():
	arr = []
	count = 0
	for f in files:	
		#increment file counter
		count+=1

		with open(os.path.join(label_src, f),'r') as anns:
			#get the next line
			lines = anns.readlines()
			
			for line in lines:
				#get the first char
				c = line[0]

				#append
				arr.append(c)

	return arr, count


#get class totals
class_array, file_count = get_class_totals()

classes, counts = np.unique(class_array, return_counts=True)
total = sum(counts)
max_class_count = np.where(max(counts))[0][0]

print('Files: {}'.format(file_count))	
print('Classesi sumary: {}'.format(dict(zip(classes, counts))))
print('Raw regions: {}'.format(total))



'''
CALCULATE region multiplier
'''
mult = []
for i in range(len(classes)):
	#ratio of max class to current class
	mult.append(int(min(counts[max_class_count]/counts[i], 10)))
	#change 1s to 0s - most common class does not need
	#more regions
	mult = [0 if x==1 else x for x in mult]

print(mult)



'''
Multiply regions
'''
for f in files:	
	with open(os.path.join(label_src, f),'r') as anns:
		#get the next line
		lines = anns.readlines()
	
		#cant read then write, so hold new_lines
		#in variable
		new_lines = []
	
		for line in lines:
			#get the first char = class number
			c = int(line[0])
			
			#get class index
			cl_index =  int(classes[c])
			
			#multiplier
			multiplier = mult[cl_index]
			
			#new coordinates
			old_xyhw = []
			new_xyhw = []

			#line marker
			marker = 0
			
			#get x,y,h,w of this line
			for k in range(4):
				#get next occurance of '.' after marker
				marker = line.find('.', marker+1)
				
				#read next float
				old_xyhw.append(float(line[marker-1:line.find(' ', marker)].strip()))
			
			#print('Old coords: {}'.format(old_xyhw))
			
			# now write the new coordinates at the 
			# end of the file
			for m in range(multiplier-1):
				#multiply origial array
				new_xyhw = [old_xyhw[0], old_xyhw[1], old_xyhw[2]+(2*X_INCREMENT * (m+1)), old_xyhw[3]+(2*Y_INCREMENT * (m+1))]
				new_line = line[0] + ' ' + str(new_xyhw[0]) + ' ' + str(new_xyhw[1]) + ' ' + str(new_xyhw[2]) + ' ' + str(new_xyhw[3]) + '\n'
				new_lines.append(new_line)
				#print(new_line)


			#print('')
	
	#now write to same file
	with open(os.path.join(label_src, f),'a') as anns_out:
		for line in new_lines:
			anns_out.write(line)


class_array, file_count = get_class_totals()
classes, counts = np.unique(class_array, return_counts=True)
print('New classes sumary: {}'.format(dict(zip(classes, counts))))
