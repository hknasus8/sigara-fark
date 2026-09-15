import streamlit as st
import cv2
import numpy as np
import os
import requests
import urllib.parse
import easyocr

st.set_page_config(
    page_title="Sigara Standı Akıllı Denetim Sistemi",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Arayüz gizleme stilleri
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppToolbar {visibility: hidden; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden; display: none !important;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# EasyOCR Okuyucuyu Belleğe Yükle (Önbellekli)
@st.cache_resource
def ocr_okuyucu_yukle():
    return easyocr.Reader(['tr', 'en'], gpu=False)

with st.spinner("AI Metin Okuma (OCR) motoru hazırlanıyor..."):
    reader = ocr_okuyucu_yukle()

def resmi_boyutlandir(img, max_genislik=1000):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_genislik:
        oran = max_genislik / float(w)
        yeni_yukseklik = int(h * oran)
        return cv2.resize(img, (max_genislik, yeni_yukseklik), interpolation=cv2.INTER_AREA)
    return img

if "app_password" not in st.secrets:
    st.error("⚠️ Kritik Güvenlik Uyarısı: 'app_password' Streamlit secrets içinde tanımlı değil!")
    st.stop()

app_pass = st.secrets["app_password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Sigara Standı Akıllı Denetim Sistemi - Giriş")
    st.markdown("<p style='color: gray; font-size: 14px; margin-top: -15px;'>Developed by Hakan</p>", unsafe_allow_html=True)
    
    sifre_input = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary"):
        if sifre_input == app_pass:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre!")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Denetim ve OCR Ayarları")
min_area_val = st.sidebar.slider("Minimum Eksik Boyutu (Hassasiyet)", 50, 2000, 150, step=25)
fark_esigi = st.sidebar.slider("Piksel Fark Eşiği (Yoğunluk)", 20, 100, 40, step=5)

YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"

@st.cache_data(ttl=600, show_spinner=False)
def yandex_sehirleri_getir(public_key):
    headers = {"User-Agent": "Mozilla/5.0"}
    sehirler = []
    try:
        root_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&limit=200"
        resp = requests.get(root_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return [], f"Kök Dizin Okunamadı (HTTP {resp.status_code})"

        root_items = resp.json().get("_embedded", {}).get("items", [])
        for item in root_items:
            if item.get("type") == "dir":
                b_name = item.get("name")
                if b_name.upper() in ["BAYİ", "BAYI"]:
                    bayi_path = item.get("path")
                    sub_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={urllib.parse.quote(bayi_path, safe='/')}&limit=200"
                    sub_resp = requests.get(sub_url, headers=headers, timeout=15)
                    if sub_resp.status_code == 200:
                        for sub_item in sub_resp.json().get("_embedded", {}).get("items", []):
                            if sub_item.get("type") == "dir":
                                sehirler.append(sub_item.get("name"))
                else:
                    sehirler.append(b_name)
        return sorted(list(set(sehirler))), None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=600, show_spinner=False)
def yandex_sehir_bayilerini_getir(public_key, sehir_adi):
    headers = {"User-Agent": "Mozilla/5.0"}
    bayiler = []
    try:
        root_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&limit=200"
        resp = requests.get(root_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return [], "Kök Dizin Okunamadı"

        root_items = resp.json().get("_embedded", {}).get("items", [])
        sehir_item_found = None
        
        for item in root_items:
            if item.get("type") == "dir" and item.get("name", "").upper() == sehir_adi.upper():
                sehir_item_found = item
                break
        
        if not sehir_item_found:
            for item in root_items:
                if item.get("type") == "dir" and item.get("name", "").upper() in ["BAYİ", "BAYI"]:
                    bayi_path = item.get("path")
                    sub_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={urllib.parse.quote(bayi_path, safe='/')}&limit=200"
                    sub_resp = requests.get(sub_url, headers=headers, timeout=15)
                    if sub_resp.status_code == 200:
                        for sub_item in sub_resp.json().get("_embedded", {}).get("items", []):
                            if sub_item.get("type") == "dir" and sub_item.get("name", "").upper() == sehir_adi.upper():
                                sehir_item_found = sub_item
                                break
                    if sehir_item_found:
                        break

        if not sehir_item_found:
            return [], f"'{sehir_adi}' klasörü bulunamadı."

        sehir_path = sehir_item_found.get("path")
        bayi_list_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={urllib.parse.quote(sehir_path, safe='/')}&limit=500"
        bayi_resp = requests.get(bayi_list_url, headers=headers, timeout=15)
        if bayi_resp.status_code != 200:
            return [], "Şehir içeriği okunamadı"

        bayi_items = bayi_resp.json().get("_embedded", {}).get("items", [])
        for b_item in bayi_items:
            if b_item.get("type") == "dir":
                b_name = b_item.get("name")
                b_path = b_item.get("path")
                if b_name and b_path:
                    bayiler.append({"name": b_name, "path": b_path})

        return sorted(bayiler, key=lambda x: x["name"]), None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=600, show_spinner=False)
def yandex_bayi_gorseli_getir(public_key, bayi_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        encoded_path = urllib.parse.quote(bayi_path, safe='/')
        api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={encoded_path}&limit=200"
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None, "Klasör içeriği okunamadı."

        final_embedded = resp.json().get("_embedded")
        if not final_embedded:
            return None, "Seçilen bayi klasörü boş."
            
        sub_items = final_embedded.get("items", [])
        gorsel_download_url = None
        for sub_item in sub_items:
            if sub_item.get("type") == "file":
                file_name = sub_item.get("name", "").lower()
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    gorsel_download_url = sub_item.get("file")
                    break

        if gorsel_download_url:
            img_resp = requests.get(gorsel_download_url, headers=headers, timeout=15)
            if img_resp.status_code == 200:
                image_bytes = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    return resmi_boyutlandir(img), None

        return None, "İçerikte uygun görsel bulunamadı."
    except Exception as e:
        return None, f"Hata: {e}"

col_baslik, col_cikis = st.columns([5, 1])
with col_baslik:
    st.title("SİGARA STANDI AKILLI DENETİM SİSTEMİ")
    st.markdown("<p style='color: gray; font-size: 14px; margin-top: -15px;'>Developed by Hakan</p>", unsafe_allow_html=True)

with col_cikis:
    st.write("") 
    if st.button("🚪 Çıkış Yap", type="secondary"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("---")
st.subheader("1. Lokasyon ve Bayi Seçimi")

dinamik_sehirler, sehir_hata = yandex_sehirleri_getir(YANDEX_ROOT_PUBLIC_KEY)
col_s1, col_s2 = st.columns(2)

with col_s1:
    secilen_sehir_adi = st.selectbox(
        "Şehir Seçin", 
        options=dinamik_sehirler if dinamik_sehirler else ["Şehir Bulunamadı"],
        index=None,
        placeholder="Lütfen bir şehir seçin..."
    )

bayiler_listesi = []
bayi_hata = None
if secilen_sehir_adi:
    bayiler_listesi, bayi_hata = yandex_sehir_bayilerini_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_sehir_adi)

with col_s2:
    if secilen_sehir_adi:
        if bayiler_listesi:
            secilen_bayi_adi = st.selectbox(
                "Bayi Seçin", 
                options=[b["name"] for b in bayiler_listesi],
                index=None,
                placeholder="Lütfen bir bayi seçin..."
            )
        else:
            secilen_bayi_adi = st.selectbox("Bayi Seçin", ["Bayi Bulunamadı"], index=None)
    else:
        secilen_bayi_adi = st.selectbox("Bayi Seçin", ["Önce Şehir Seçmelisiniz"], index=None, disabled=True)

secilen_bayi_path = ""
if secilen_sehir_adi and secilen_bayi_adi and bayiler_listesi and secilen_bayi_adi != "Bayi Bulunamadı":
    for b in bayiler_listesi:
        if b["name"] == secilen_bayi_adi:
            secilen_bayi_path = b["path"]
            break

if st.button("🔄 Önbelleği Yenile"):
    yandex_sehirleri_getir.clear()
    yandex_sehir_bayilerini_getir.clear()
    yandex_bayi_gorseli_getir.clear()
    st.toast("Önbellek temizlendi!", icon="🔄")
    st.rerun()

st.markdown("---")

ref_img = None
hata_mesaji = None
if secilen_bayi_path:
    with st.spinner(f"'{secilen_bayi_adi}' için Yandex Disk'ten görsel yükleniyor..."):
        ref_img, hata_mesaji = yandex_bayi_gorseli_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_bayi_path)

col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("2. Referans (İdeal) Görsel")
    if secilen_bayi_path and ref_img is not None:
        st.success(f"✅ Referans Yüklendi")
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.info("ℹ️ Şehir ve bayi seçin.")

with col_up2:
    st.subheader("3. Sahadan Gelen Görsel")
    if secilen_bayi_path:
        curr_file = st.file_uploader("Fotoğraf yükleyin", type=["jpg", "jpeg", "png"], key="curr")
        curr_img = None
        if curr_file is not None:
            curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
            raw_curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)
            curr_img = resmi_boyutlandir(raw_curr_img)
            st.success("✅ Fotoğraf yüklendi")
            st.image(curr_img, channels="BGR", use_container_width=True)
    else:
        st.file_uploader("Fotoğraf yükleyin", type=["jpg", "jpeg", "png"], key="curr_disabled", disabled=True)

st.markdown("---")
st.subheader("4. Stand Kapasite Ayarı")
ideal_urun_sayisi = st.number_input("Standda Bulunması Gereken Toplam Ürün (Slot) Sayısı", min_value=1, value=50, step=1)
st.markdown("---")

if "result_img" not in st.session_state:
    st.session_state.result_img = None
if "eksik_sayisi" not in st.session_state:
    st.session_state.eksik_sayisi = 0
if "raf_yuzdesi" not in st.session_state:
    st.session_state.raf_yuzdesi = 100.0
if "ocr_raporu" not in st.session_state:
    st.session_state.ocr_raporu = []
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False

if secilen_sehir_adi and secilen_bayi_adi and ref_img is not None and 'curr_file' in locals() and curr_file is not None and 'curr_img' in locals() and curr_img is not None:
    if st.button("Hassas Görsel ve Etiket (OCR) Analizini Başlat", type="primary"):
        with st.spinner("Stand arka planı, renk mimarisi ve raf yapısı doğrulanıyor..."):
            
            # --- YENİ MİMARİ: GÖRSEL İSKELET, TON VE RENK KONTROLÜ ---
            img_h, img_w = ref_img.shape[:2]
            curr_img_resized = cv2.resize(curr_img, (img_w, img_h), interpolation=cv2.INTER_AREA)

            # 1. Renk Dağılımı ve Ton Analizi (Siyah iskelet vs Beyaz/Gri iskelet)
            hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
            hsv_curr = cv2.cvtColor(curr_img_resized, cv2.COLOR_BGR2HSV)
            
            hist_ref = cv2.calcHist([hsv_ref], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
            
            hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
            
            renk_benzerligi = cv2.compareHist(hist_ref, hist_curr, cv2.HISTCMP_CORREL)

            # 2. Stand Mimarisi (Arka Plan ve Raflar) - Ürünleri blurla, sadece iskeleti bırak
            gray_ref_blur = cv2.GaussianBlur(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), (55, 55), 0)
            gray_curr_blur = cv2.GaussianBlur(cv2.cvtColor(curr_img_resized, cv2.COLOR_BGR2GRAY), (55, 55), 0)
            
            yapi_farki_matrisi = cv2.absdiff(gray_ref_blur, gray_curr_blur)
            ortalama_yapi_farki = np.mean(yapi_farki_matrisi)

            stand_tipi_uyusuyor = True
            hata_mesaji_detayi = ""

            # Karar Verme: Renk profili çok farklıysa veya iskelet parlaklıkları (Siyah vs Beyaz) uymuyorsa hata ver
            if renk_benzerligi < 0.20:
                stand_tipi_uyusuyor = False
                hata_mesaji_detayi = "Genel renk mimarisi (Örn: Siyah iskelet vs. Beyaz iskelet) tamamen farklı."
            elif ortalama_yapi_farki > 35:
                stand_tipi_uyusuyor = False
                hata_mesaji_detayi = "Standın raf sayısı, boşlukları veya arka plan materyali yapısal olarak uyuşmuyor."

            # Yapı ve Renk testini geçenler için son bir OCR güvenlik adımı (Marka kontrolü)
            if stand_tipi_uyusuyor:
                ref_ust = ref_img[0:int(img_h * 0.20), :]
                curr_ust = curr_img_resized[0:int(img_h * 0.20), :]
                
                ref_ocr = reader.readtext(ref_ust)
                curr_ocr = reader.readtext(curr_ust)
                
                ref_kelimeler = [r[1].strip().lower() for r in ref_ocr if len(r[1].strip()) >= 4]
                tabela_kelimeleri = [k for k in ref_kelimeler if k in ["nqkta", "tobacco", "nokta", "market", "tekel"]]
                
                if tabela_kelimeleri:
                    curr_kelimeler = " ".join([c[1].strip().lower() for c in curr_ocr])
                    eslesme_var = any(tk in curr_kelimeler for tk in tabela_kelimeleri)
                    if not eslesme_var:
                        stand_tipi_uyusuyor = False
                        hata_mesaji_detayi = "Referans standın tabelası sahadaki görselde okunamadı/bulunamadı."

            if not stand_tipi_uyusuyor:
                st.error(f"🚨 **UYARI: Stand Tipi ve Yapı Uyuşmazlığı!** \n\n Sahadan Gelen Görsel, Referans Görsel ile yapısal olarak eşleşmiyor. \n\n**Tespit Edilen Sebep:** {hata_mesaji_detayi} \n\n Lütfen doğru standa ait bir fotoğraf yükleyin.")
                st.session_state.analiz_yapildi = False
            else:
                # --- Stand Uygunsa 1. Aşama: Kırmızı Kutu Analizi Başlar ---
                curr_img = curr_img_resized # İşlemlere eşitlenmiş boyuttan devam et
                
                gray_ref = cv2.GaussianBlur(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                gray_curr = cv2.GaussianBlur(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

                diff = cv2.absdiff(gray_ref, gray_curr)
                _, thresh = cv2.threshold(diff, fark_esigi, 255, cv2.THRESH_BINARY)
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
                morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = []
                
                for c in contours:
                    if cv2.contourArea(c) > min_area_val:
                        x, y, w, h = cv2.boundingRect(c)
                        if (img_w * 0.01 < x < img_w * 0.99) and (img_h * 0.02 < y < img_h * 0.98):
                            boxes.append([x, y, x + w, y + h])

                def non_max_suppression(boxes, overlapThresh=0.15):
                    if not boxes: return []
                    boxes = np.array(boxes)
                    pick = []
                    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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
                
                # --- 2. Aşama: EasyOCR ile Etiket / Ürün Adı Okuma ve Kıyaslama ---
                ref_ocr_results = reader.readtext(ref_img)
                curr_ocr_results = reader.readtext(curr_img)
                
                mismatch_details = []
                result_img = curr_img.copy()
                eksik_sayisi = len(filtered_boxes)
                hesaplanan_yuzde = max(0.0, 100.0 - ((eksik_sayisi / max(1, ideal_urun_sayisi)) * 100.0))

                for idx, (startX, startY, endX, endY) in enumerate(filtered_boxes, 1):
                    cv2.rectangle(result_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(result_img, f"Eksik #{idx}", (startX + 3, startY + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                ref_texts = [r[1].strip().lower() for r in ref_ocr_results if len(r[1].strip()) > 2]

                for (c_box, c_text, c_prob) in curr_ocr_results:
                    clean_c_text = c_text.strip().lower()
                    if len(clean_c_text) > 2:
                        eslesti = any(r_t in clean_c_text or clean_c_text in r_t for r_t in ref_texts)
                        
                        if not eslesti:
                            mismatch_details.append(c_text)
                            pts = np.array(c_box, dtype=np.int32)
                            cv2.polylines(result_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                            
                            pt_x = int(c_box[0][0])
                            pt_y = int(c_box[0][1] - 5)
                            cv2.putText(result_img, "Etiket Uyumsuz", (pt_x, max(15, pt_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                cv2.rectangle(result_img, (0, 0), (img_w, 100), (0, 0, 0), -1)
                cv2.putText(result_img, f"Bayi: {secilen_bayi_adi}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_img, f"Eksik: {eksik_sayisi} | Uyusmayan Etiket: {len(mismatch_details)}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if eksik_sayisi > 0 else (0, 255, 0), 2)

                st.session_state.result_img = result_img
                st.session_state.eksik_sayisi = eksik_sayisi
                st.session_state.raf_yuzdesi = hesaplanan_yuzde
                st.session_state.ocr_raporu = mismatch_details
                st.session_state.analiz_yapildi = True

    if st.session_state.analiz_yapildi and st.session_state.result_img is not None:
        st.subheader("Tespit Edilen Eksikler ve Etiket (OCR) Karşılaştırma Raporu")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("📊 Raf Doğruluk Oranı", f"%{st.session_state.raf_yuzdesi:.1f}")
        col_m2.metric("⚠️ Eksik/Boşluk Alan (Kırmızı)", f"{st.session_state.eksik_sayisi} Adet")
        col_m3.metric("🔵 Uyuşmayan Etiket (Mavi)", f"{len(st.session_state.ocr_raporu)} Adet")

        if st.session_state.ocr_raporu:
            with st.expander("🔍 Etiket Uyuşmazlık Detayları (Mavi Kutularla İşaretlenenler)"):
                st.write("Sahada okunan ancak referans görsel ile uyuşmayan veya şüpheli etiket yazıları:")
                for text in set(st.session_state.ocr_raporu):
                    st.markdown(f"- 🔵 `{text}`")
        else:
            st.success("✅ Tüm ürün ve etiket isimleri referans görsel ile tam uyumlu!")

        sonuc_gorsel_genisligi = st.slider("🔍 Sonuç Görseli Boyutunu Ayarla", 300, 2000, 800, step=100)
        st.image(st.session_state.result_img, channels="BGR", width=sonuc_gorsel_genisligi)

        success, encoded_image = cv2.imencode(".jpg", st.session_state.result_img)
        if success:
            st.download_button(
                label="📥 Rapor Fotoğrafını İndir",
                data=encoded_image.tobytes(),
                file_name=f"{secilen_bayi_adi.replace(' ', '_')}_analiz_sonucu.jpg",
                mime="image/jpeg"
            )
else:
    st.info("ℹ️ Analiz için şehir, bayi seçin ve sahadan gelen fotoğrafı yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
