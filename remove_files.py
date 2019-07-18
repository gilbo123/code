import subprocess
import os

folders = os.listdir('/media/gilbert/Ann/')

for f in folders:
	bashCommand = 'cd '+f+'; rm -R *\(1\).jpg'
	output = subprocess.run(bashCommand, shell=True)
	print(f)
	print(output)


