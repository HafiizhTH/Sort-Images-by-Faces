import streamlit as st
import numpy as np
from PIL import Image
from preprocessing import preprocess_image
from face_detect import detect_faces
from embeddings import get_face_encodings
from qdrant_db import create_collection, store_embedding, search_embedding
from matching import match_uploaded_image

# Panggil create_collection() saat aplikasi pertama kali dijalankan
create_collection()

def main():
    st.title("Face Recognition App")

    # Halaman 1: Upload gambar ke database
    menu = ["Upload Multiple Images", "Match Uploaded Image"]
    choice = st.sidebar.selectbox("Select Activity", menu)

    if choice == "Upload Multiple Images":
        st.subheader("Upload Multiple Images")
        uploaded_files = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png", "heic"], accept_multiple_files=True)
        
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
                
                # Proses gambar
                image_array = np.array(image)
                face_locations = detect_faces(image_array)
                
                if face_locations:
                    face_encodings = get_face_encodings(image_array, face_locations)
                    store_embedding("face_embeddings", uploaded_file.name, face_encodings[0])
                else:
                    store_embedding("non_face_embeddings", uploaded_file.name, [])

            st.success(f'{len(uploaded_files)} file berhasil di-upload!')

        # Tampilkan gambar di dalam database (contoh)
        st.subheader("All Images in Database")
        # Logika untuk menampilkan semua gambar dalam database di sini
        
    elif choice == "Match Uploaded Image":
        st.subheader("Match Uploaded Image")
        uploaded_file = st.file_uploader("Unggah Gambar Wajah", type=["jpg", "jpeg", "png", "heic"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Konversi gambar untuk diproses
            image_array = np.array(image)
            
            # Cek gambar dengan database
            matched_images = match_uploaded_image(image_array)
            
            if matched_images:
                st.write('Found matches:')
                for img_path in matched_images:
                    st.image(img_path, width=100)
            else:
                st.write('No matches found.')

if __name__ == '__main__':
    main()
