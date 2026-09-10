import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sigara Standı Akıllı Denetim Sistemi",
    page_icon="🚬",
    layout="wide"
)

# Kenar çubuğuna logo ekleme
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", use_container_width=True)
elif os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

st.title("SİGARA STANDI AKILLI DENETİM SİSTEMİ - Bayi Bazlı Fark Analizi")
st.markdown("---")

# Kenar çubuğu ayarları
st.sidebar.header("Denetim ve Bayi Seçimi")
threshold_val = st.sidebar.slider("Fark Hassasiyet Eşiği", 10, 100, 30)

# Örnek bayi listesi
bayi_listesi = ["Bayi_001_Ahmet_Market", "Bayi_002_Mehmet_Tekel", "Bayi_003_Can_Büfe"]
secilen_bayi = st.sidebar.selectbox("Denetlenecek Bayiyi Seçin", bayi_listesi)

# Kontrol edilecek mevcut görseli yükleme
st.subheader(f"Seçilen Bayi: {secilen_bayi}")
curr_file = st.file_uploader("Kontrol Edilecek (Mevcut) Stand Görselini Yükleyin", type=["jpg", "jpeg", "png"])

if curr_file is not None:
    curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
    curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mevcut Görsel")
        st.image(curr_img, channels="BGR", use_container_width=True)

    if st.button("Seçilen Bayiyi Analiz Et", type="primary"):
        ref_img = curr_img.copy() 

        with st.spinner(f"{secilen_bayi} için karşılaştırma yapılıyor..."):
            gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray_ref, gray_curr)
            _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_img = curr_img.copy()
            eksik_sayisi = 0
            report_data = []
            
            for i, c in enumerate(contours):
                if cv2.contourArea(c) > 400: 
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    eksik_sayisi += 1
                    report_data.append({
                        "Bayi": secilen_bayi,
                        "Fark ID": eksik_sayisi,
                        "Konum": f"X: {x}, Y: {y}",
                        "Durum": "Eksik / Değişiklik"
                    })

            with col2:
                st.subheader("Analiz Sonucu ve Tespitler")
                st.image(result_img, channels="BGR", use_container_width=True)

                success, encoded_image = cv2.imencode(".jpg", result_img)
                if success:
                    st.download_button(
                        label="📥 Sonuç Fotoğrafını İndir (JPG)",
                        data=encoded_image.tobytes(),
                        file_name=f"{secilen_bayi}_analiz.jpg",
                        mime="image/jpeg"
                    )

        if eksik_sayisi > 0:
            st.error(f"{secilen_bayi} denetimi tamamlandı: {eksik_sayisi} farklılık tespit edildi.")
        else:
            st.success(f"{secilen_bayi} denetimi tamamlandı: Stand ideal durumda.")
            
        if report_data:
            st.dataframe(report_data, use_container_width=True)
else:
    st.info("Lütfen sol menüden bayiyi seçip, yukarıdan mevcut durumu gösteren görseli yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
