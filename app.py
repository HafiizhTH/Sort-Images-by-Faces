import streamlit as st
import os
import numpy as np
from PIL import Image
import pillow_heif
import io  # Perlu untuk memproses file yang diunggah dari memori
import model  # Mengimpor file model.py yang berisi logika face recognition

# Fungsi untuk menampilkan gambar di grid
def display_images(image_paths, header=None):
    if header:
        st.subheader(header)
    
    if image_paths:
        cols = st.columns(4)  # Buat grid untuk menampilkan gambar
        for idx, image_path in enumerate(image_paths):
            try:
                # Cek apakah gambar adalah .heic dan gunakan pillow_heif untuk membukanya
                if image_path.lower().endswith('.heic'):
                    heif_file = pillow_heif.read_heif(image_path)
                    image = Image.frombytes(
                        heif_file.mode, 
                        heif_file.size, 
                        heif_file.data
                    )
                else:
                    # Baca gambar biasa dengan Pillow
                    image = Image.open(image_path)
                
                # Tampilkan gambar di Streamlit
                with cols[idx % 4]:
                    st.image(image, use_column_width=True)
            except Exception as e:
                st.warning(f"Gambar tidak dapat dibuka: {image_path}. Error: {str(e)}")

# Fungsi utama untuk Streamlit app
def main():
    st.title("Aplikasi Pencocokan Wajah")

    # Component upload image
    uploaded_file = st.file_uploader("Unggah Gambar Wajah", type=["jpg", "jpeg", "png", "heic"])
    
    # Jika pengguna telah mengunggah gambar
    if uploaded_file is not None:
        try:
            # Membaca file .HEIC jika diunggah oleh pengguna
            if uploaded_file.name.lower().endswith(".heic"):
                heif_file = pillow_heif.read_heif(io.BytesIO(uploaded_file.read()))
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data
                )
            else:
                # Membaca gambar biasa dari memori
                image = Image.open(uploaded_file)

            st.success("Gambar berhasil diunggah dan sedang diproses!")

            # Tampilkan pesan sementara "Sedang memproses gambar..."
            processing_message = st.empty()
            processing_message.text("Sedang memproses gambar...")

            # Proses pencocokan wajah langsung dari memori
            matched_images = model.match_images(np.array(image), "photo")
            
            # Ubah pesan menjadi "Model selesai memproses gambar"
            processing_message.text("Model selesai memproses gambar")

            if matched_images:
                st.success(f"Ditemukan {len(matched_images)} gambar yang cocok:")
                display_images(matched_images)
            else:
                st.warning("Tidak ditemukan gambar yang cocok.")
                st.text("Gambar-gambar yang ada di folder tidak memiliki kecocokan dengan gambar yang Anda unggah.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {str(e)}")

    # Jika tidak ada gambar yang di-upload, tampilkan semua gambar di folder 'photo'
    else:
        image_paths = [os.path.join("photo", f) for f in os.listdir("photo") if model.is_image(f)]
        display_images(image_paths, "Gambar di Folder")

if __name__ == '__main__':
    main()
