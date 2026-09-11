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

# Sağ üstteki Share, GitHub ve üst menü çubuğunu gizleyen CSS stilleri
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppToolbar {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Kenar çubuğuna logo ekleme
if os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", width=220)
elif os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=220)

st.sidebar.markdown("---")

# Kenar çubuğu ayarları
st.sidebar.header("Denetim ve Bayi Seçimi")
threshold_val = st.sidebar.slider("Boşluk / Eksik Hassasiyeti", 50, 200, 110)

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

# Üst kısım: Referans Görsel (Yandex) ve Sahadan Gelen (Manuel Yükleme) Yan Yana
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.subheader("1. Referans (İdeal) Görsel")
    if ref_img is not None:
        st.success(f"✅ '{secilen_bayi}' Yandex Disk'ten Yüklendi")
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.warning(f"⚠️ '{secilen_bayi}' için Yandex Disk'te klasör veya görsel bulunamadı.")

with col_up2:
    st.subheader("2. Kontrol Edilecek (Mevcut) Görsel")
    curr_file = st.file_uploader("Sahadan gelen fotoğrafı yükleyin", type=["jpg", "jpeg", "png"], key="curr")
    
    curr_img = None
    if curr_file is not None:
        curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
        curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)
        st.success("✅ Sahadan gelen foto yüklendi")
        st.image(curr_img, channels="BGR", use_container_width=True)

st.markdown("---")

# Eğer hem referans görsel (Yandex'ten) hem de sahadan gelen görsel hazırsa analiz bölümünü aç
if ref_img is not None and curr_file is not None and curr_img is not None:
    # Boyutları eşitleme
    if ref_img.shape != curr_img.shape:
        curr_img = cv2.resize(curr_img, (ref_img.shape[1], ref_img.shape[0]))

    if st.button("Farkı Analiz Et ve Eksikleri Bul", type="primary"):
        with st.spinner("Standlardaki boşluklar ve eksikler taranıyor..."):
            
            # Gri tonlamaya çevir
            gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            # İki görsel arasındaki genel yapısal hizalamayı oturtmak için ORB tabanlı feature matching (Opsiyonel kaydırma telafisi)
            # Doğrudan raftaki koyu renkli boşlukları (ürün olmayan arka plan alanlarını) yakalama mantığı:
            # Stand raflarındaki ürünler renkli/paketlidir, eksik yerler ise ahşap arka plan veya koyu gölgedir.
            
            # Referans ile mevcut görselin mutlak farkı yerine, mevcut görselin kendi içindeki koyu/boş alan analizi + referansla kıyas
            diff = cv2.absdiff(gray_ref, gray_curr)
            
            # Aydınlatma farklarını elemek için blur ve adaptif eşikleme
            diff_blur = cv2.GaussianBlur(diff, (15, 15), 0)
            _, thresh = cv2.threshold(diff_blur, threshold_val, 255, cv2.THRESH_BINARY)

            # Sadece raftaki dikey/yatay ürün bloklarına denk gelen büyük eksik alanları filtrele
            kernel = np.ones((9, 9), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in contours:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                
                # Stand raflarının yüksekliğine ve paket boyutlarına uygun filtre (Çok küçük gürültüleri ve tüm ekranı ele)
                if 1500 < area < 80000 and h > 30 and w > 30:
                    # Sadece üst raflar ve orta raflardaki ürün alanlarını sınırla (Gereksiz zeminleri alma)
                    if y > ref_img.shape[0] * 0.15: 
                        boxes.append([x, y, x + w, y + h])

            # Non-Maximum Suppression (Üst üste binen kutuları tekilleştirme)
            def non_max_suppression(boxes, overlapThresh=0.1):
                if len(boxes) == 0:
                    return []
                boxes = np.array(boxes)
                pick = []
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                idxs = np.argsort(y2)
                
                while len(idxs) > 0:
                    last = len(idxs) - 1
                    i = idxs[last]
                    pick.append(i)
                    
                    xx1 = np.maximum(x1[i], x1[idxs[:last]])
                    yy1 = np.maximum(y1[i], y1[idxs[:last]])
                    xx2 = np.minimum(x2[i], x2[idxs[:last]])
                    yy2 = np.minimum(y2[i], y2[idxs[:last]])
                    
                    w = np.maximum(0, xx2 - xx1 + 1)
                    h = np.maximum(0, yy2 - yy1 + 1)
                    
                    overlap = (w * h) / area[idxs[:last]]
                    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
                    
                return boxes[pick].astype("int")

            filtered_boxes = non_max_suppression(boxes)

            result_img = curr_img.copy()
            eksik_sayisi = len(filtered_boxes)
            
            # Kutuların içine sıra numarasını yazdırma
            for idx, (startX, startY, endX, endY) in enumerate(filtered_boxes, 1):
                cv2.rectangle(result_img, (startX, startY), (endX, endY), (0, 0, 255), 3)
                cv2.putText(result_img, f"#{idx}", (startX + 8, startY + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            st.subheader("Tespit Edilen Gerçek Eksikler / Boşluklar")
            st.image(result_img, channels="BGR", use_container_width=True)

            success, encoded_image = cv2.imencode(".jpg", result_img)
            if success:
                st.download_button(
                    label="📥 Sonuç Fotoğrafını İndir",
                    data=encoded_image.tobytes(),
                    file_name=f"{secilen_bayi.replace(' ', '_')}_analiz_sonucu.jpg",
                    mime="image/jpeg"
                )

        if eksik_sayisi > 0:
            st.error(f"{secilen_bayi} denetimi tamamlandı: Toplam {eksik_sayisi} adet eksik/boş alan tespit edildi ve numaralandırıldı.")
        else:
            st.success(f"{secilen_bayi} denetimi tamamlandı: Stand düzeninde belirgin bir eksik veya boşluk bulunamadı.")
else:
    st.info("ℹ️ Sol tarafta Yandex Disk'ten gelen referans görseli görebilirsiniz. Analiz yapabilmek için lütfen sağ taraftan **Sahadan Gelen Fotoğrafı** yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
