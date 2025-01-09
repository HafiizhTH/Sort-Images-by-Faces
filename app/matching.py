from face_detect import detect_faces
from embeddings import get_face_encodings
from qdrant_db import search_embedding

def match_uploaded_image(uploaded_image_array):
    face_locations = detect_faces(uploaded_image_array)
    face_encodings = get_face_encodings(uploaded_image_array, face_locations)
    
    if not face_encodings:
        return []

    query_vector = face_encodings[0].tolist()
    
    # Mencari gambar yang paling cocok dari collection "face_embeddings"
    search_result = search_embedding(query_vector)
    
    matched_images = [result.payload['path'] for result in search_result]
    return matched_images
