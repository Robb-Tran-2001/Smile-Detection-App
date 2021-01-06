import cv2

#cCapture webcam video
webcam = cv2.VideoCapture(0)

#Create face and smile detectors
face_detector = cv2.CascadeClassifier('faces.xml')
smile_detector = cv2.CascadeClassifier('smiles.xml')

while True:
    #Read the frame
    (success, frame) = webcam.read()
    if not success:
        continue

    #If successful, convert to grayscale    
    frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Get all faces in the frame.
    faces = face_detector.detectMultiScale(frame_gs)
    
    #For each face
    for (x, y, w, h) in faces:
        
        #Draw green rectangle over it
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        #Get frame of face, turn into grayscale (only scan for smiles in face frames)
        face_frame = frame[y:y+h, x:x+w]
        face_gs = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        
        #Get all smiles, if more than 1, write text underneath
        #Use scalefactor to blur for less pixels, more optimized.
        #Min neighbors means at least 20 neighboring rectangles to count as a face
        smiles = smile_detector.detectMultiScale(face_gs, scaleFactor=1.7, minNeighbors=20)
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3,
                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
        
        #Render. Press Q to end
        cv2.imshow('Why so serious?', frame)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break

#Release
webcam.release()
cv2.destroyAllWindows()


print("Code completed")