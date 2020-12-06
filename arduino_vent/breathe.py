import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


'''
CONSTANTS
'''
######### PLOT RELATED #########
#resolution for plot
resolution = 1000

#time in seconds for plot
seconds = 10

################################



#breaths per minute
breaths_per_min = 30 # must be >= 6 

#max volume of air
max_volume = 1000  # mL

#time of inpiratory pause
ins_pause = 100  # ms

#time of expiratory pause
exp_pause = 200 # ms 

#I/E ratio
IE = (1, 3) # = in:out 





#inspiratory percent 
insp_percent = IE[0] / (IE[0]+ IE[1])

#expiratory percent
exp_percent = IE[1] / (IE[0] + IE[1])

print('INS : EXP (%) = {} : {}'.format(insp_percent, exp_percent))



'''
CALCS
'''
######### PLOT RELATED #########
#total breaths (for plotting)
num_breaths = seconds * breaths_per_min / 60
print('num breaths: {}'.format(num_breaths))

#seconds per breath 
ms_per_breath = seconds * 1000 / num_breaths
print('ms per breath: {}'.format(ms_per_breath))

# increment range = 0-seconds
RANGE = np.linspace(0, seconds, seconds*resolution)

#increments per second
inc_per_ms = len(RANGE) / seconds * 1000
print('inc per sec: {}'.format(inc_per_ms/1000))

# increments per breath
inc_per_breath = ms_per_breath / 1000 * inc_per_ms / 1000

#increments of inspiratory pause (increments in 1 sec * time)
ins_pause_inc = int(inc_per_ms / 1000 * ins_pause / 1000)
print('pause increment : {}'.format(ins_pause_inc))

#increments of inspiratory pause (increments in 1 sec * time)
exp_pause_inc = int(inc_per_ms / 1000 * exp_pause / 1000)

#breath indexes
insp_range = int((inc_per_breath * insp_percent) - ins_pause_inc) #I/E - pause
exp_range = int((inc_per_breath * exp_percent) - exp_pause_inc) #(1-I/E) - pause

print('Inspiratory range: {}'.format(insp_range))

#volume increment variables
insp_vol_inc = max_volume / insp_range

exp_vol_inc = max_volume / exp_range


################################




print('Range: {}'.format(len(RANGE)))
print('inc per breath: {}'.format(inc_per_breath))

'''
LOOP
'''
#value array (volume/increment)
X = []

#sys.exit()

#graph insantiate
plt.style.use("ggplot")
plt.figure()


#counters
breath = 0
vol = 0
#for each breath
for inc in range(len(RANGE)):
    #FULL breaths
    if inc % int(inc_per_breath) == 0:
        breath += 1
        print('Breath: {}'.format(breath))
   
        #breathe in range without pause
        for ins in range(insp_range):#max_volume):
            #increment volume counter
            vol += insp_vol_inc
            #print(vol)
            #append to array
            X.append(vol)
            #X.append(max_volume - max_volume * math.exp(-vol * 0.006))

        #add pause
        for in_p in range(ins_pause_inc):
            X.append(vol)

        #breathe out range without pause
        for exp in range(exp_range):#max_volume):
            #increment volume counter
            vol -= exp_vol_inc
            #append to array
            X.append(vol)
        
        
        #add pause
        for ex_p in range(exp_pause_inc):
            X.append(vol)

#add extra values if X too short
if len(X) < len(RANGE):
    for ex in range(len(RANGE) - len(X)):
        #check if greater than max

        #increment
        vol += 1
        X.append(vol)

#clip end of array if too long
if len(X) > len(RANGE):
    X = X[:len(RANGE)]

#sys.exit()
#print('X: {}'.format(X))



#plot graph
N = np.arange(0, len(RANGE))
plt.plot(N, X, label="Volume")
#plt.plot(N, VOL, label="volume")
plt.title("Breathing pattern")
plt.xlabel("Time (ms)")
plt.ylabel("Volume (mL)")
#print(dir(plt))
#plt.legend(loc="upper right")
#plt.savefig('loss_plot.png')
plt.show()

