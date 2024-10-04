"""
import cv2
cap = cv2.VideoCapture("C:\\Users\\PTC-SW3\\Downloads\\sample.mp4")
print("cap",cap)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(700,450))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",frame)
    cv2.imshow("gray",gray)
    k = cv2.waitKey(250)
    if k == ord("q") :
        break
cap.release()
cv2.destroyAllWindows() """

import cv2
import pafy
url="https://www.youtube.com/watch?v=ZAOZxISntbY&ab_channel=Tracklib"
data = pafy.new(url)
data = pafy.getbest(preftype = "mp4")
cap = cv2.VideoCapture(0)
cap.open(data.ulr)
print("cap",cap)

# output save the video
#fourcc = cv2.VideoWriter_fourcc(*"XVID")
#output = cv2.VideoWriter("output.mp4",fourcc,20.0,(640,480))

while (cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame,(700,450))
        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame",frame)
        #cv2.imshow("gray",gray)
        #output.write(frame)
        k = cv2.waitKey(1)
        if k == ord("q") :
            break
cap.release()
#output.release()
cv2.destroyAllWindows()