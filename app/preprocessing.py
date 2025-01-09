import cv2

# Fungsi untuk melakukan preprocessing gambar (resize dan penghapusan noise)
def preprocess_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path)
    
    # Resize gambar
    resized_image = cv2.resize(image, size)
    
    # Menghilangkan noise menggunakan Gaussian Blur
    processed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    return processed_image
