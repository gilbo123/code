import cv2
import os

path = './IMAGES/foreign/'
out = './utils/output/'

files = os.listdir(path)

for f in files:
    fn = path + f
    img = cv2.imread(fn)
    print(img.shape)

    img = cv2.resize(img, (50, 50))
    img = cv2.medianBlur(img, 7)
    print(img.shape)
    cv2.imshow('image', img)
    k = cv2.waitKey(100)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    
    cv2.imwrite(out + f, img)
   
cv2.destroyAllWindows()

