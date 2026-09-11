import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import requests

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

# Yandex Disk 'BAYİ' Klasörünün Public Linki
YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/JXJNYBDAk6fePw"

# Excel dosyasından bayileri okuma
excel_dosya_adi = "bayiler.xlsx"
bayi_listesi = []

if os.path.exists(excel_dosya_adi):
    try:
        df_bayiler = pd.read_excel(excel_dosya_adi, sheet_name="DATA")
        if "UNVAN" in df_bayiler.columns:
            bayi_listesi = df_bayiler["UNVAN"].dropna().astype(str).tolist()
        else:
            bayi_listesi = df_bayiler.iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        st.sidebar.error(f"Excel okunurken hata oluştu: {e}")

if not bayi_listesi:
    bayi_listesi = ["Excel dosyasından unvanlar okunamadı"]

secilen_bayi = st.sidebar.selectbox("Denetlenecek Bayiyi Seçin", bayi_listesi)

st.title("SİGARA STANDI AKILLI DENETİM SİSTEMİ - Fark Analizi")
st.markdown(f"**Seçilen Bayi:** {secilen_bayi}")
st.markdown("<p style='color: gray; font-size: 14px;'>Developed by Hakan</p>", unsafe_allow_html=True)
st.markdown("---")

# Yandex Disk'ten esnek eşleşme ile bayi klasörünü ve görseli bulan fonksiyon
def yandex_bayi_gorseli_getir(public_key, bayi_adi):
    try:
        api_url = f"https://cloud-api.yandex.net:443/v1/disk/public/resources?public_key={public_key}&limit=2000"
        resp = requests.get(api_url)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        items = data.get("_embedded", {}).get("items", [])
        
        hedef_aranan = bayi_adi.strip().lower()
        bayi_klasor_path = None
        
        for item in items:
            if item.get("type") == "dir":
                Item_Adi = item.get("name", "").strip().lower()
                if hedef_aranan in Item_Adi or Item_Adi in hedef_aranan:
                    bayi_klasor_path = item.get("path")
                    break
        
        if not bayi_klasor_path:
            return None
            
        sub_api_url = f"https://cloud-api.yandex.net:443/v1/disk/public/resources?public_key={public_key}&path={bayi_klasor_path}&limit=100"
        sub_resp = requests.get(sub_api_url)
        if sub_resp.status_code != 200:
            return None
            
        sub_items = sub_resp.json().get("_embedded", {}).get("items", [])
        
        gorsel_download_url = None
        for sub_item in sub_items:
            if sub_item.get("type") == "file":
                file_name = sub_item.get("name", "").lower()
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    gorsel_download_url = sub_item.get("file")
                    break
                    
        if gorsel_download_url:
            img_resp = requests.get(gorsel_download_url)
            if img_resp.status_code == 200:
                image_bytes = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
                return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Yandex bağlantı hatası: {e}")
    return None

# Referans görseli Yandex Disk'ten otomatik çekme
ref_img = None
with st.spinner(f"'{secilen_bayi}' için Yandex Disk'te arama yapılıyor..."):
    ref_img = yandex_bayi_gorseli_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_bayi)

# Görsel yükleme alanları
col_up1, col_up2 = st.columns(2)

with col_up1:
    if ref_img is not None:
        st.success(f"✅ '{secilen_bayi}' Referans Görseli Yandex'ten Otomatik Yüklendi")
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.warning("⚠️ Yandex Disk'te bu bayiye ait klasör veya fotoğraf bulunamadı. Lütfen manuel yükleyin:")
        ref_file = st.file_uploader("1. Referans (İdeal) Stand Görseli (Manuel)", type=["jpg", "jpeg", "png"], key="ref")
        if ref_file is not None:
            ref_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
            ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)

with col_up2:
    st.info("Kontrol edilecek mevcut sahadaki fotoğrafı yükleyin:")
    curr_file = st.file_uploader("2. Kontrol Edilecek (Mevcut) Görsel", type=["jpg", "jpeg", "png"], key="curr")
    
    curr_img = None
    if curr_file is not None:
        curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
        curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)
        st.success("✅ Sahadan gelen foto yüklendi")
        st.image(curr_img, channels="BGR", use_container_width=True)

st.markdown("---")

# Eğer referans görsel ve mevcut görsel hazırsa analiz ekranını başlat
if ref_img is not None and curr_file is not None and curr_img is not None:
    if ref_img.shape != curr_img.shape:
        curr_img = cv2.resize(curr_img, (ref_img.shape[1], ref_img.shape[0]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Referans Görsel")
        st.image(ref_img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("Sahadan Gelen")
        st.image(curr_img, channels="BGR", use_container_width=True)

    if st.button("Farkı Analiz Et ve Eksikleri Bul", type="primary"):
        with st.spinner("Görseller karşılaştırılıyor ve farklar hesaplanıyor..."):
            gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray_ref, gray_curr)
            _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_img = curr_img.copy()
            eksik_sayisi = 0
            
            for i, c in enumerate(contours):
                if cv2.contourArea(c) > 400: 
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    eksik_sayisi += 1

            with col3:
                st.subheader("Tespit Edilen Farklar")
                st.image(result_img, channels="BGR", use_container_width=True)

                success, encoded_image = cv2.imencode(".jpg", result_img)
                if success:
                    st.download_button(
                        label="📥 Farkları Gösteren Fotoğrafı İndir",
                        data=encoded_image.tobytes(),
                        file_name=f"{secilen_bayi.replace(' ', '_')}_analiz_sonucu.jpg",
                        mime="image/jpeg"
                    )

        if eksik_sayisi > 0:
            st.error(f"{secilen_bayi} denetimi tamamlandı: Toplam {eksik_sayisi} farklılık / eksik bölge kırmızı çerçeveyle işaretlendi.")
        else:
            st.success(f"{secilen_bayi} denetimi tamamlandı: Referans görsel ile mevcut görsel arasında belirgin bir fark bulunamadı.")
else:
    st.info("Lütfen sol menüden bayiyi seçin (Yandex Disk'ten fotoğraf otomatik gelecektir) ve sağdan **2. Kontrol Edilecek Görseli** yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
