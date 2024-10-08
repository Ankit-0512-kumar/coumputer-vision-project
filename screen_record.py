import cv2 as c
import numpy as np
import pyautogui as pg

# Create resolution
rs = pg.size()

# filename in which we store recording
fn = input("enter path")
# fix the frame rate
fps = 40.0

fourcc = c.VideoWriter_fourcc(*"XVID")
output = c.VideoWriter(fn,fourcc,fps,rs)

# recording
c.namedWindow("Live_recording", c.WINDOW_NORMAL)
c.resizeWindow("Live_recording",(600,400))

while True:
    img = pg.screenshot()
    f = np.array(img)
    f = c.cvtColor(f,c.COLOR_BGR2RGB)
    output.write(f)
    c.imshow("Live_recording",f)
    if c.waitKey(1) == ord("q"):
        break

output.release()
c.destroyAllWindows()
