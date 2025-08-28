import cv2
import numpy as np
import face_recognition
import os

images = []
class_names = []
#takes images from images folder and creates list
image_list = os.listdir("images")
print(image_list)
#for each image in the image list
for cl in image_list:
    current_image = cv2.imread(f"images/{cl}") #store it as a variable
    images.append(current_image) #add raw image to images list
    class_names.append(os.path.splitext(cl)[0]) #add the name of the image to class_names list
print(class_names)

#encodes a list of images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converts from bgr to rgb
        encode_img = face_recognition.face_encodings(img)[0] #encodes img
        encode_list.append(encode_img)
    return encode_list

#stores encoded versions of images stored under the images folder
current_encodings = find_encodings(images)
print("encoding complete")

#Enable webcam
#creates a VideoCapture object that activates the webcam
#the zero means default webcam
cap = cv2.VideoCapture(0)

#the infinite loop runs the camera until stopped
while True:
    #captures one frame from the camera
    #sucess tells us if the camera worked(gives a boolean)
    #img gives us the image data from the camera
    success, img = cap.read()
    #resizes the img
    img_small = cv2.resize(
        img, #input image from camera
        (0, 0), # Target size in pixels - (0,0) means "ignore this, use scaling factors instead"
        None, # Interpolation method (None = use default)
        0.25, #controls horizontal scaling factor: fx
        0.25 #controls vertical scaling factor: fy
        )
    #converts to rgb from bgr
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(img_small) #Gets faces from image
    # Scale face locations back to original image size
    faces_original_size = []
    for (top, right, bottom, left) in faces_current_frame:
        faces_original_size.append((top*4, right*4, bottom*4, left*4))
    encode_img = face_recognition.face_encodings(img, faces_original_size) #encodes img using the faces_current_frame for efficiency
    
    #takes a single face from faces_current frame and stores that as face_location
    #Then encodes that face using encode_img and stores that as encode_face
    for encode_face, face_location in zip(encode_img, faces_original_size):
        #compares the list of images stored in the images folder to the encoded face on the webcam
        matches = face_recognition.compare_faces(current_encodings, encode_face)
        #This will check the face distance of each image in images folder.
        face_distance = face_recognition.face_distance(current_encodings, encode_face)
        print(face_distance)
        #finds index one with lowest distance value(best match)
        match_index = np.argmin(face_distance)

        y1, x1, y2, x2 = face_location
        def box(red, green, blue, condition = False):
            cv2.rectangle(
            img, 
            (x1, y1), 
            (x2, y2),
            (blue, green, red), #color
            2 #thickness
            )
            cv2.rectangle(
                img,
                (x1, y2 - 35),
                (x2, y2),
                (blue, green, red),
                cv2.FILLED
            )

            displayed_name = ""
            if condition:
                displayed_name = name
            else:
                displayed_name = "Unkown"

            # Calculate a reasonable font size based on face size
            face_width = abs(x2 - x1)
            font_scale = max(0.5, min(1.5, face_width / 300))

            cv2.putText(
                img,
                displayed_name,
                (x2 + 10, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                (255, 255, 255),
                2
            )


        if matches[match_index]:
            name = class_names[match_index].title()
            print(name)
            box(0, 255, 0, True)
        else:
            print("Unknown")
            box(255, 0, 0)
    
    #shows webcam
    cv2.imshow('Webcam', img)
    #controls framerate (60fps)
    cv2.waitKey(1)