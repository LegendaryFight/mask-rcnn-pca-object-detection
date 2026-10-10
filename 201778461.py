import cv2
import math
import random
import time
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

#configuration settings
SCORE_THRESHOLD=0.7          #minimum confidence score for detections
MASK_THRESHOLD=0.5           #threshold for converting predicted mask to binary mask
AXIS_SCALE=120.0             #maximum axis length for PCA visualization
CAMERA_INDEX=0               #default webcam
SAVE_VIDEO=True              #save processed output to video file
OUTPUT_VIDEO_PATH="201778461.mp4"
SHOW_WINDOW=False            #keep False for environments without GUI support
MAX_RECORD_SECONDS=45        #maximum recording time in real life seconds

#reduce frame size to speed up CPU inference
FRAME_WIDTH=640
FRAME_HEIGHT=480

#use a low playback FPS so the saved video looks normal
OUTPUT_VIDEO_FPS=1.0

#coco class labels
COCO_NAMES=[
    "__background__","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
    "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

#assign a stable random color to each class for consistent visualization
random.seed(42)
CLASS_COLORS=[[random.randint(50,255),random.randint(50,255),random.randint(50,255)] for _ in range(len(COCO_NAMES))]

#device and model loading
def get_device():
    #use GPU if available, otherwise CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device):
    #load pretrained Mask R-CNN from torchvision
    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model=torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.eval()#set model to evaluation mode
    model.to(device)
    return model

#helper functions
def get_class_name(label_id):
    #safe class lookup to avoid index errors
    if 0<=label_id<len(COCO_NAMES):
        return COCO_NAMES[label_id]
    return f"class_{label_id}"

def get_class_color(label_id):
    #safe color lookup even if an unexpected label appears
    return CLASS_COLORS[label_id%len(CLASS_COLORS)]

def overlay_mask(image,mask,color,alpha=0.45):
    #draw a semi-transparent colored mask over the detected object
    colored=np.zeros_like(image,dtype=np.uint8)
    colored[:]=color
    binary=mask.astype(bool)
    image[binary]=cv2.addWeighted(image[binary],1.0-alpha,colored[binary],alpha,0)
    return image

#Q4 center of segmented mask
def compute_mask_center(mask):
    #compute center using only foreground mask pixels, not the bounding box
    ys,xs=np.where(mask>0)
    if len(xs)==0 or len(ys)==0:
        return None

    cx=float(np.mean(xs))
    cy=float(np.mean(ys))
    return (cx,cy)

#Q4 PCA on segmented mask
def compute_principal_axes(mask):
    #extract all foreground pixel coordinates from the binary mask
    ys,xs=np.where(mask>0)
    if len(xs)<5:
        return None,None,None

    #convert coordinates into 2D point set for PCA
    points=np.column_stack((xs,ys)).astype(np.float32)

    #apply PCA using OpenCV
    mean,eigenvectors,eigenvalues=cv2.PCACompute2(points,mean=None)

    #PCA mean is the centroid of the mask pixels
    center=(float(mean[0,0]),float(mean[0,1]))

    #first eigenvector = major principal axis
    #second eigenvector = minor principal axis
    v1=eigenvectors[0]
    v2=eigenvectors[1]

    #eigenvalues indicate spread/variance along each axis
    l1=float(eigenvalues[0,0])
    l2=float(eigenvalues[1,0])

    return center,(v1,l1),(v2,l2)

def draw_axes(image,center,axis1,axis2,color1=(255,0,0),color2=(0,255,255)):
    #draw the two principal axes at the mask center
    cx=int(round(center[0]))
    cy=int(round(center[1]))

    v1,l1=axis1
    v2,l2=axis2

    #scale axis lengths according to eigenvalues
    s1=max(40.0,min(AXIS_SCALE,math.sqrt(max(l1,1.0))*8.0))
    s2=max(30.0,min(AXIS_SCALE,math.sqrt(max(l2,1.0))*8.0))

    #endpoints for major axis
    p1=(int(round(cx+v1[0]*s1)),int(round(cy+v1[1]*s1)))
    p2=(int(round(cx-v1[0]*s1)),int(round(cy-v1[1]*s1)))

    #endpoints for minor axis
    p3=(int(round(cx+v2[0]*s2)),int(round(cy+v2[1]*s2)))
    p4=(int(round(cx-v2[0]*s2)),int(round(cy-v2[1]*s2)))

    #draw axes
    cv2.line(image,(cx,cy),p1,color1,3,cv2.LINE_AA)
    cv2.line(image,(cx,cy),p2,color1,3,cv2.LINE_AA)
    cv2.line(image,(cx,cy),p3,color2,3,cv2.LINE_AA)
    cv2.line(image,(cx,cy),p4,color2,3,cv2.LINE_AA)

    #draw center point
    cv2.circle(image,(cx,cy),5,(0,0,255),-1,cv2.LINE_AA)

def draw_label(image,text,x1,y1,color):
    #draw label text with a colored background for readability
    font=cv2.FONT_HERSHEY_SIMPLEX
    scale=0.55
    thickness=2
    (tw,th),baseline=cv2.getTextSize(text,font,scale,thickness)
    y_text=max(y1-10,th+10)
    cv2.rectangle(image,(x1,y_text-th-8),(x1+tw+8,y_text+baseline-4),color,-1)
    cv2.putText(image,text,(x1+4,y_text-4),font,scale,(0,0,0),thickness,cv2.LINE_AA)

#Q3 and Q4 frame processing
def process_frame(frame,model,device):
    #resize frame to speed up model inference on CPU
    frame=cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))

    #convert BGR image from OpenCV to RGB for PyTorch
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    tensor=F.to_tensor(rgb).to(device)

    #run Mask R-CNN inference
    with torch.no_grad():
        output=model([tensor])[0]

    #copy frame for drawing visualizations
    vis=frame.copy()

    #extract model outputs
    boxes=output["boxes"].detach().cpu().numpy()
    labels=output["labels"].detach().cpu().numpy()
    scores=output["scores"].detach().cpu().numpy()
    masks=output["masks"].detach().cpu().numpy()

    #process each detected object
    for i in range(len(boxes)):
        score=float(scores[i])

        #skip low-confidence detections
        if score<SCORE_THRESHOLD:
            continue

        label_id=int(labels[i])
        class_name=get_class_name(label_id)
        color=get_class_color(label_id)

        #bounding box coordinates
        box=boxes[i].astype(int)
        x1,y1,x2,y2=box.tolist()

        #convert predicted mask probability to binary mask
        mask_prob=masks[i,0]
        mask=(mask_prob>MASK_THRESHOLD).astype(np.uint8)

        #ignore empty masks
        if np.count_nonzero(mask)==0:
            continue

        #draw segmentation mask overlay
        vis=overlay_mask(vis,mask,color,alpha=0.4)

        #draw contour around mask to make edges clearer
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis,contours,-1,color,2)

        #draw bounding box
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)

        #draw label and confidence
        label_text=f"{class_name}: {score:.2f}"
        draw_label(vis,label_text,x1,y1,color)

        #Q4 compute center directly from segmented mask
        center=compute_mask_center(mask)

        #Q4 compute PCA principal axes from mask pixels
        pca_center,axis1,axis2=compute_principal_axes(mask)

        #draw PCA axes if available, otherwise draw only center point
        if pca_center is not None and axis1 is not None and axis2 is not None:
            draw_axes(vis,pca_center,axis1,axis2)
        elif center is not None:
            cx=int(round(center[0]))
            cy=int(round(center[1]))
            cv2.circle(vis,(cx,cy),5,(0,0,255),-1,cv2.LINE_AA)

    #display processing device without FPS
    info=f"Device: {device.type.upper()}"
    cv2.putText(vis,info,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(vis,info,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1,cv2.LINE_AA)

    #overlay explanation text so the video is self-explanatory
    cv2.putText(vis,"Mask R-CNN + PCA",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(vis,"Center from mask | Axes via PCA",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)

    return vis

#main program
def main():
    device=get_device()
    print(f"Using device: {device}")

    #load model once at startup
    model=load_model(device)

    #open webcam
    cap=cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    #force lower capture resolution to reduce CPU load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)

    writer=None
    if SAVE_VIDEO:
        #use low playback FPS so the saved video duration is not extremely short
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        writer=cv2.VideoWriter(OUTPUT_VIDEO_PATH,fourcc,OUTPUT_VIDEO_FPS,(FRAME_WIDTH,FRAME_HEIGHT))

    print("Recording started.")
    print(f"SHOW_WINDOW={SHOW_WINDOW}")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")
    print(f"Max record time: {MAX_RECORD_SECONDS} seconds")
    print(f"Output video FPS: {OUTPUT_VIDEO_FPS}")

    start_time=time.time()
    frame_count=0

    while True:
        ret,frame=cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        #process current frame through Mask R-CNN and PCA pipeline
        result=process_frame(frame,model,device)
        frame_count+=1

        #optional live window for systems that support GUI
        if SHOW_WINDOW:
            cv2.imshow("Mask R-CNN Webcam + PCA Principal Axes",result)
            key=cv2.waitKey(1)&0xFF
            if key==ord("q"):
                print("Stopped by user.")
                break

        #save processed frame into output video
        if writer is not None:
            writer.write(result)

        #stop after the chosen real-time duration
        elapsed=time.time()-start_time
        if elapsed>=MAX_RECORD_SECONDS:
            print("Reached max record time.")
            break

        #print progress every few processed frames
        if frame_count%5==0:
            print(f"Processed {frame_count} frames | elapsed {elapsed:.1f}s")

    #release resources
    cap.release()
    if writer is not None:
        writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("Recording finished.")
    print(f"Saved video to: {OUTPUT_VIDEO_PATH}")

if __name__=="__main__":
    main()