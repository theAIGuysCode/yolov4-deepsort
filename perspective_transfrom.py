import cv2
import sys
import numpy as np

video_center_point=np.array([940,397],dtype='float32')
video_points=np.array([[1431, 397], [450, 397], [940, 97], [940, 1022]],dtype='float32')
row,col=video_points.shape
for i in range(row):
    video_points[i]-=video_center_point

soccer_field_center=np.array([512,327],dtype='float32')
soccer_field_points=np.array([[594,327],[429,327],[512,26],[512,626]],dtype='float32')
row,col=soccer_field_points.shape
for i in range(row):
    soccer_field_points[i]-=soccer_field_center

soccer_filed_img=cv2.imread('data/video/Soccer_field.png')
M=cv2.getPerspectiveTransform(video_points,soccer_field_points)
#print(M)

#input a tuple return a tuple
def perspective_transform(people_point):
    (x,y)=people_point
    people_point=np.array([x,y],dtype='float32')
    people_point-=video_center_point
    people_point=np.r_[people_point,1] #insert in row
    #print(people_point.shape)

    dst_people_point=np.dot(M,people_point)
    dst_people_point=dst_people_point[0:2]
    dst_people_point+=soccer_field_center
    (x,y)=(int(dst_people_point[0]),int(dst_people_point[1]))
    return (x,y)
    #print((x,y))

if __name__=="__main__":
    #dst_people_point=cv2.perspectiveTransform(people_point.T,M)
    cv2.namedWindow('Perspective',cv2.WINDOW_NORMAL)
    soccer_filed_img=cv2.imread('data/video/Soccer_field.png')
    print(soccer_filed_img.shape)
    (x,y)=perspective_transform((1276,284)) #a simpl people example, remember to input the foot point
    cv2.circle(soccer_filed_img,(x,y),5,(255,0,0),thickness=3)
    while 1:
        cv2.imshow('Perspective',soccer_filed_img)
        # k=cv2.waitKey(1)&0xFF #the wait key must put here
        # if not cv2.getWindowProperty('Perspective',cv2.WND_PROP_VISIBLE):
        #     break
        if cv2.waitKey(0)&0xFF==27:
            break
        cv2.destroyAllWindows()


