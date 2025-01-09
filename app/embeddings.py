import face_recognition

def get_face_encodings(image, face_locations):
    # Mendapatkan embedding wajah
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings
