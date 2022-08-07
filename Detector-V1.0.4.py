import cv2
import os

os.chdir('models')

def detectFace(net, frame, confidence_threshold=0.7):
    # taking the frame
    frameDNN = frame.copy()
    # print(frameDNN.shape)  #(height, width, channel) 
    frameHeight = frameDNN.shape[0]
    frameWidth = frameDNN.shape[1]

    # preprocessing - scale factor, size, mean substraction, swapRB ie., BGR channel, crop
    blob = cv2.dnn.blobFromImage(frameDNN, 1.0, (227,227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    
    net.setInput(blob) # giving input in the net
    detections = net.forward() # procedding for the neural network
    # print(detections) # get 4D array of 7 things
    # 0, 1. confidence, x1, y1, x2, y2
    
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2] 
        if confidence > confidence_threshold:
            # draw bounding box
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            # drawing rectangle
            cv2.rectangle(frameDNN, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameDNN, faceBoxes

# calling pre trained model
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

# define the output categories
genderList = ['Male', 'Female'] 
ageList = ['(0-2)', '(4-6)', '(8-14)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# define model
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# video capturing
video = cv2.VideoCapture(0) # 0 - inbuilt webcam , 1 - external camera
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg, faceBoxes = detectFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
    
    for faceBox in faceBoxes:
        # padding
        face = frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        # preprocess the face
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        
        # input to gendernet and agenet
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        #print(gender)

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        #print(age)

        # define the bounding box text specifications like fonts, thickness, color etc
        cv2.putText(resultImg, f'{gender},{age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()