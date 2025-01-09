import numpy as np
import face_recognition
from PIL import Image
import cv2
import pillow_heif
import os
import glob

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

# Fungsi untuk melakukan preprocessing gambar (resize dan penghapusan noise)
def preprocess_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path)
    
    # Resize gambar
    resized_image = cv2.resize(image, size)
    
    # Menghilangkan noise menggunakan Gaussian Blur
    processed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    return processed_image

# Fungsi untuk mendeteksi wajah menggunakan face_recognition
def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations

def get_images_with_faces_in_selected_folder(path='./'):
    images_with_faces = []
    for entry in glob.glob(path + '/*'):
        if not is_image(entry) or not os.path.isfile(entry):
            continue
            
        # Preprocessing sebelum face detection
        preprocessed_image = preprocess_image(entry)
        face_locations = detect_faces(preprocessed_image)
        
        # Pastikan ada wajah yang terdeteksi
        if face_locations is not None and len(face_locations) > 0:
            images_with_faces.append({"entry": entry, "image_binary": preprocessed_image, "face_locations": face_locations})
    return images_with_faces

def find_matching_images(sample_image_path, folder_path):
    # Preprocessing untuk sample image
    preprocessed_sample = preprocess_image(sample_image_path)
    sample_face_encodings = face_recognition.face_encodings(preprocessed_sample)

    # Pastikan ada wajah yang terdeteksi di gambar sampel
    if not sample_face_encodings:
        print("Tidak ada wajah yang ditemukan di gambar sampel.")
        return []

    sample_encoding = sample_face_encodings[0]

    # Load gambar dari folder dan deteksi wajah
    images_with_faces = get_images_with_faces_in_selected_folder(folder_path)
    
    matching_images = []
    
    # Periksa kecocokan dengan wajah di folder
    for image_info in images_with_faces:
        face_encodings_in_image = face_recognition.face_encodings(image_info['image_binary'], image_info['face_locations'])
        
        for face_encoding in face_encodings_in_image:
            # Bandingkan wajah sampel dengan wajah dari gambar di folder
            matches = face_recognition.compare_faces([sample_encoding], face_encoding, tolerance=0.5)
            if True in matches:
                matching_images.append(image_info['entry'])
                break

    return matching_images

if __name__ == '__main__':
    sample_image_path = "./testing/4.jpg"
    folder_path = "./training"
    
    # Validasi format gambar sampel
    if not is_image(sample_image_path):
        print("Format file gambar tidak sesuai. Harap gunakan format gambar yang ditentukan.")
    else:
        matching_images = find_matching_images(sample_image_path, folder_path)

        if matching_images:
            print("Gambar yang cocok ditemukan:")
            for match in matching_images:
                print(match)
        else:
            print("Tidak ditemukan gambar yang cocok.")
