import face_recognition
import os
import cv2

def load_known_faces(known_faces_folder):
    known_faces = []
    known_names = []

    for filename in os.listdir(known_faces_folder):
        image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
        known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

def verify_user(known_faces_folder):
    known_faces, known_names = load_known_faces(known_faces_folder)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # Find face locations in the current frame
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            for face_encoding in face_encodings:
                # Check if the face matches any of the known faces
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Draw a rectangle around the detected face and display the name
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Video', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_faces_folder = "known_faces"  # Replace with the folder containing known faces images
    verify_user(known_faces_folder)
