import os
import glob
import cv2
import numpy as np
import face_recognition
from PIL import Image
import pillow_heif

def is_image(file_name):
    image_extensions = ['png', 'jpg', 'jpeg', 'heic']  # Menambahkan dukungan untuk format .heic
    return file_name.split('.')[-1].lower() in image_extensions

# Fungsi untuk membaca gambar, termasuk konversi .heic menjadi format yang dapat diproses
def read_image(image_path):
    extension = image_path.split('.')[-1].lower()
    
    if extension == 'heic':
        # Menggunakan pillow_heif untuk membaca file .heic
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data
        )
        # Mengonversi ke format yang dapat diolah OpenCV (misal: RGB ke BGR)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # Baca gambar biasa menggunakan OpenCV
        return cv2.imread(image_path)

# Fungsi untuk melakukan preprocessing gambar (resize dan penghapusan noise)
def preprocess_image(image_path, size=(256, 256)):
    image = read_image(image_path)
    
    if image is None:
        return None
    
    # Resize gambar
    resized_image = cv2.resize(image, size)
    
    # Menghilangkan noise menggunakan Gaussian Blur
    processed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    return processed_image

# Fungsi untuk mendeteksi wajah menggunakan face_recognition
def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations

# Fungsi untuk mendapatkan gambar-gambar dengan wajah dari folder yang dipilih
def get_images_with_faces_in_selected_folder(path='./'):
    images_with_faces = []
    for entry in glob.glob(path + '/*'):
        if not is_image(entry) or not os.path.isfile(entry):
            continue
            
        # Preprocessing sebelum face detection
        preprocessed_image = preprocess_image(entry)
        if preprocessed_image is None:
            continue
        
        face_locations = detect_faces(preprocessed_image)
        
        # Pastikan ada wajah yang terdeteksi
        if face_locations is not None and len(face_locations) > 0:
            images_with_faces.append({"entry": entry, "image_binary": preprocessed_image, "face_locations": face_locations})
    return images_with_faces

# Fungsi untuk menemukan gambar yang cocok dari folder berdasarkan gambar sampel
def match_images(sample_image_array, folder_path):
    # Preprocessing untuk sample image
    preprocessed_sample = cv2.resize(sample_image_array, (256, 256))
    sample_face_encodings = face_recognition.face_encodings(preprocessed_sample)
    
    if not sample_face_encodings:
        return []

    sample_encoding = sample_face_encodings[0]

    # Load gambar dari folder dan deteksi wajah
    images_with_faces = get_images_with_faces_in_selected_folder(folder_path)
    
    matching_images = []
    
    # Periksa kecocokan dengan wajah di folder
    for image_info in images_with_faces:
        face_encodings_in_image = face_recognition.face_encodings(image_info['image_binary'], image_info['face_locations'])
        
        for face_encoding in face_encodings_in_image:
            matches = face_recognition.compare_faces([sample_encoding], face_encoding, tolerance=0.5)
            if True in matches:
                matching_images.append(image_info['entry'])
                break

    return matching_images

