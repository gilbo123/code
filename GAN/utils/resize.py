import cv2
import os

path = '/home/gil/Documents/IMAGES/GAN/over/'
out = '/home/gil/Documents/IMAGES/GAN/TRAIN/'

files = os.listdir(path)

for f in files:
    fn = path + f
    img = cv2.imread(fn)
    print(img.shape)

    img = cv2.resize(img, (48, 48))
    img = cv2.medianBlur(img, 9)
    print(img.shape)
    cv2.imshow('image', img)
    k = cv2.waitKey(33)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    
    cv2.imwrite(out + f, img)
   
cv2.destroyAllWindows()

