import cv2 as cv
import sys

ref_points=[]

def get_point(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        print("x={},y={}".format(x,y))
        xy="%d,%d"%(x,y)
        ref_points.append((x,y))
        cv.circle(frame, (x, y), 1, (255, 0, 0), thickness = 1)
        cv.putText(frame,xy,(x,y),cv.FONT_HERSHEY_PLAIN,2.0,(0,0,0),thickness=2)
        cv.imshow('Soccer',frame)


file_name=sys.argv[1].split('.')
if file_name[1]=='mp4':
    video_capture=cv.VideoCapture(sys.argv[1])
    fps = int(video_capture.get(cv.CAP_PROP_FPS))
    width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print("fps:", fps)
    print("width:", width)
    print("height:", height)
    ret_val,frame=video_capture.read()
    if ret_val:
        cv.namedWindow('Soccer',cv.WINDOW_NORMAL)
        cv.setMouseCallback('Soccer',get_point)
    else:
        print("error get frame")
elif file_name[1]=='png' or file_name[1]=='jpg':
    frame=cv.imread(sys.argv[1])
    cv.namedWindow('Soccer',cv.WINDOW_NORMAL)
    cv.setMouseCallback('Soccer',get_point)

def extract_points():
    while 1:
        cv.imshow('Soccer',frame)
        k=cv.waitKey(1)&0xFF #the wait key must put here
        if not cv.getWindowProperty('Soccer',cv.WND_PROP_VISIBLE):
            break
        if k==27:
            break
    cv.destroyAllWindows()
    return ref_points

#cmd example
#python extract_pixel_pos.py data/video/id3_500f.mp4
#
if __name__=="__main__":
    extract_points()
    print(ref_points)
