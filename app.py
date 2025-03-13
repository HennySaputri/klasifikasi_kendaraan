import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os 
import tempfile
import shutil

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE =False

# Konfigurasi halaman
st.set_page_config(page_title="Pengenalan Gambar Kendaraan")

# Periksa apakah library YOLO tersedia
def cek_library():
    if not YOLO_AVAILABLE:
        st.error("Ultralytics tidak terpasang. Silakan instal dengan perintah berikut:")
        st.code("pip install ultralytics")
        return False
    return True


st.markdown(""""
    <div style="background-color:#BF3131; padding: 20px; text-align: center;">
        <h1 style="color: white;">Program Pengenalan Gambar</h1>
        <h5 style="color: white;">Deteksi Gambar Kendaraan</h5>
    </div>
""", unsafe_allow_html=True)

# Pastikan library sudah terpasang sebelum melanjutkan
if cek_library():
    # upload gambar
     uploaded_file = st.file_uploader("upload gambar kendaraan", type=['jpg', 'jpeg', 'png'])

     if uploaded_file:
        # Simpan sementara
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "gambar.jpg")
        image = Image.open(uploaded_file)

        # Ubah ukuran gambar agar lebih konsisten
        image = image.resize((300, 300))
        image.save(temp_file)

        # Tampilkan gambar dengan batasan ukuran CSS
        st.markdown("<div style='text-align:center;'>",unsafe_allow_html=True )
        st.image(image, caption="Gambar yang diupload")
        st.markdown("</div>", unsafe_allow_html=True)

        # Deteksi gambar
        if st.button("Deteksi Gambar"):
            with st.spinner("Sedang diproses..."):
                try:
                    model = YOLO('best.pt')
                    hasil = model(temp_file)

                    # Ambil hasil prediksi
                    nama_objek = hasil[0].names
                    nilai_prediksi = hasil[0].probs.data.numpy().tolist()
                    objek_terdeteksi = nama_objek[np.argmax(nilai_prediksi)]

                    # Buat grafik
                    grafik = go.Figure([go.Bar(x=list(nama_objek.values()), y=nilai_prediksi)])
                    grafik.update_layout(title='Tingkat Keyakinan Prediksi', xaxis_title='Kendaraan', yaxis_title='Keyakinan')

                    # Tampilkan hasil
                    st.write(f"Kendaraan terdeteksi: {objek_terdeteksi}")
                    st.plotly_chart(grafik)

                except Exception as e:
                    st.error("Gambar tidak dapat dikenali.")
                    st.error(f"Error: {e}")

                # Hapus file sementara setelah digunakan
                shutil.rmtree(temp_dir, ignore_errors=True)

# Footer dengan gaya CSS
st.markdown(
    "<div style='text-align: center;' class='footer'>Program Aplikasi deteksi kendaraan @2025</div>",
    unsafe_allow_html=True  
)        

