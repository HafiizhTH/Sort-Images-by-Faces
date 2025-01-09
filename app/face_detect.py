import numpy as np
import face_recognition
from PIL import Image
import cv2
import pillow_heif

# Fungsi untuk membaca gambar, termasuk konversi .heic
def read_image(image_path):
    extension = image_path.split('.')[-1].lower()
    
    if extension == 'heic':
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data
        )
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(image_path)

# Deteksi wajah menggunakan face_recognition
def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations
