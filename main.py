import cv2
import numpy as np
import face_recognition

#converts bgr to rgb values
img_cena = face_recognition.load_image_file('images/cena.jpg')
img_cena = cv2.cvtColor(img_cena, cv2.COLOR_BGR2RGB)
img_cena_test = face_recognition.load_image_file('images/cena_test.jpg')
img_cena_test = cv2.cvtColor(img_cena_test, cv2.COLOR_BGR2RGB)

#Tracks location of the face
face_location = face_recognition.face_locations(img_cena)[0]
#Creates 128x64 measurements of face for comparison
encode_cena = face_recognition.face_encodings(img_cena)[0]
#Creates a box around the face using the rectangular coords of face_location
cv2.rectangle(img_cena, 
              (face_location[3], face_location[0]), 
              (face_location[1], face_location[2]),
              (255, 0, 255), #color
              2 #thickness
            )

#Tracks locations of face
face_location_test = face_recognition.face_locations(img_cena_test)[0]
#Creates 128x64 measurements of face for comparison
encode_cena_test = face_recognition.face_encodings(img_cena_test)[0]
#Creates a box around the face using the rectangular coords of face_location_test
cv2.rectangle(
    img_cena_test, 
    (face_location_test[3], face_location_test[0]), 
    (face_location_test[1], face_location_test[2]),
    (255, 0, 255), #color
    2 #thickness
    )

#compares faces
results = face_recognition.compare_faces(
    [encode_cena], #list of known faces that will be compared to test
    encode_cena_test #the face we are trying to confirm
    )
#Measures the distance between the two faces.  Lower number means better match
face_dis = face_recognition.face_distance(
    [encode_cena],
    encode_cena_test
    )
# prints results of the comparison
print(results, face_dis)

#Adds boolean result with face distance value on image test
cv2.putText(
    img_cena_test, 
    f"{results} {round(face_dis[0], 2)}", 
    (50, 50), 
    cv2.FONT_HERSHEY_COMPLEX, 
    1, 
    (0, 0, 255)
    )

#displays images
cv2.imshow("John Cena", img_cena)
cv2.imshow("John Cena Test", img_cena_test)
cv2.waitKey(0)