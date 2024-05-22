def get_speed(dist,frame1,frame2,fps):
    t=frame2-frame1
    speed=(dist*fps)/t
    return speed

def init_array(n,k):
    arr=[]
    j=0
    while(j<n):
        arr.append(k)
        j=j+1
    return arr
    


mfps = int(vid.get(cv2.CAP_PROP_FPS))
max_v=100000
enter_frame=init_array(max_v,-1)
exit_frame=init_array(max_v,-1)
mf=init_array(max_v,0)
max_tid=-1
dist=0.5
line1=500
line2=1000

#use prev
#################################


tid=track.track_id
ymin=bbox[1]
if(tid<max_v):
    if(mf[tid]==0 and ymin>=line1):
        mf[tid]=1
        enter_frame[tid]=frame_num
    if(mf[tid]==1 and ymin>=line2):
        mf[tid]=2
        exit_frame[tid]=frame_num
        max_tid=max(max_tid,tid)
            
###################################

    
j=0
while(j<max_v and j<=max_tid):
    if(mf[tid]==2):
        speed=get_speed(dist,enter_frame[tid],exit_frame[tid],mfps)