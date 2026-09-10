import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sigara Standı Akıllı Denetim Sistemi",
    page_icon="🚬",
    layout="wide"
)

# Kenar çubuğuna logo ekleme
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", width=220)
elif os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=220)

st.sidebar.markdown("---")

# Kenar çubuğu ayarları
st.sidebar.header("Denetim ve Bayi Seçimi")
threshold_val = st.sidebar.slider("Fark Hassasiyet Eşiği", 10, 100, 30)

# Excel dosyasından bayileri dinamik olarak okuma
excel_dosya_adi = "bayiler.xlsx"

bayi_listesi = []
if os.path.exists(excel_dosya_adi):
    try:
        # DATA sekmesinden 'UNVAN' sütununu okuyoruz
        df_bayiler = pd.read_excel(excel_dosya_adi, sheet_name="DATA")
        if "UNVAN" in df_bayiler.columns:
            bayi_listesi = df_bayiler["UNVAN"].dropna().astype(str).tolist()
        else:
            bayi_listesi = df_bayiler.iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        st.sidebar.error(f"Excel okunurken hata oluştu: {e}")

# Eğer liste boşsa uyarı ver
if not bayi_listesi:
    bayi_listesi = ["Excel dosyasından unvanlar okunamadı"]

secilen_bayi = st.sidebar.selectbox("Denetlenecek Bayiyi Seçin", bayi_listesi)

st.title("SİGARA STANDI AKILLI DENETİM SİSTEMİ - Fark Analizi")
st.markdown(f"**Seçilen Bayi:** {secilen_bayi}")
st.markdown("<p style='color: gray; font-size: 14px;'>Developed by Hakan</p>", unsafe_allow_html=True)
st.markdown("---")

# İki ayrı görsel yükleme alanı: Referans ve Mevcut Durum
col_up1, col_up2 = st.columns(2)
with col_up1:
    ref_file = st.file_uploader("1. Referans (İdeal) Stand Görseli", type=["jpg", "jpeg", "png"], key="ref")
with col_up2:
    curr_file = st.file_uploader("2. Kontrol Edilecek (Mevcut) Görsel", type=["jpg", "jpeg", "png"], key="curr")

st.markdown("---")

# Eğer her iki görsel de yüklendiyse analiz ekranını aç
if ref_file is not None and curr_file is not None:
    # Görselleri belleğe okuma
    ref_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
    ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)

    curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
    curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)

    # Boyut uyumsuzluğu varsa mevcut görseli referansa göre boyutlandır
    if ref_img.shape != curr_img.shape:
        curr_img = cv2.resize(curr_img, (ref_img.shape[1], ref_img.shape[0]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Referans Görsel")
        st.image(ref_img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("Mevcut Görsel")
        st.image(curr_img, channels="BGR", use_container_width=True)

    if st.button("Farkı Analiz Et ve Eksikleri Bul", type="primary"):
        with st.spinner("Görseller karşılaştırılıyor ve farklar hesaplanıyor..."):
            # Gri tonlamaya dönüştürme
            gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            # İki görsel arasındaki mutlak farkı bulma
            diff = cv2.absdiff(gray_ref, gray_curr)
            
            # Eşikleme (Thresholding) ile gürültüleri ayıklama
            _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

            # Konturları (farklı alanları) tespit etme
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Mevcut görsel üzerine tespit edilen farkları kutu içine alma
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
                        "Konum (X, Y)": f"X: {x}, Y: {y}",
                        "Durum": "Eksik / Değişiklik Tespit Edildi"
                    })

            with col3:
                st.subheader("Tespit Edilen Farklar")
                st.image(result_img, channels="BGR", use_container_width=True)

                # İşlenmiş görseli JPG olarak indirme butonu
                success, encoded_image = cv2.imencode(".jpg", result_img)
                if success:
                    st.download_button(
                        label="📥 İşlenmiş Fotoğrafı İndir (JPG)",
                        data=encoded_image.tobytes(),
                        file_name=f"{secilen_bayi.replace(' ', '_')}_analiz_sonucu.jpg",
                        mime="image/jpeg"
                    )

        if eksik_sayisi > 0:
            st.error(f"{secilen_bayi} denetimi tamamlandı: Toplam {eksik_sayisi} farklılık / eksik bölge kırmızı çerçeveyle işaretlendi.")
        else:
            st.success(f"{secilen_bayi} denetimi tamamlandı: Referans görsel ile mevcut görsel arasında belirgin bir fark bulunamadı.")
        
        # Detaylı Rapor Tablosu
        st.subheader("Denetim Raporu Detayı")
        if report_data:
            st.dataframe(report_data, use_container_width=True)
else:
    st.info("Lütfen analiz yapabilmek için yukarıdan hem **1. Referans (İdeal) Stand Görselini** hem de **2. Kontrol Edilecek (Mevcut) Görseli** yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
