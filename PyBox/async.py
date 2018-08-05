from multiprocessing import Process, Lock
import time
import cv2

def f(l, i):
    while(True):
        #l.acquire()
        time.sleep(.3)
        print 'hello world', i
        #l.release()


if __name__ == '__main__':
    lock = Lock()
    proc1 = Process(target=f, args=(lock, 1)).start()
    proc2 = Process(target=f, args=(lock, 2)).start()
    proc3 = Process(target=f, args=(lock, 3)).start()
    proc4 = Process(target=f, args=(lock, 4)).start()
    
    while True:
        #time.sleep(0.1)
        k = cv2.waitKey(10) & 0xFF
        #print(k)
        if k is not 255:
            print(k)
            print(ord(k))
        # press 'q' to exit
        if k == ord('q'):
            print('Quitting...')
            proc1.terminate()
            time.sleep(0.1)
            proc2.terminate()
            time.sleep(0.1)
            proc3.terminate()
            time.sleep(0.1)
            proc4.terminate()
            break
