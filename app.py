# -*- coding: utf-8 -*-
"""
SIGARA STANDI AKILLI DENETIM SISTEMI - Streamlit Web Sürümü
---------------------------------------------------------
Tkinter bağımlılıkları kaldırılarak Streamlit web arayüzüne 
uyarlanmıştır. v4.0 çekirdek mantığı birebir korunmaktadır[cite: 1].
"""

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ===========================================================
# AYARLAR (tum esik degerleri tek yerden yonetiliyor)
# ===========================================================
class Ayarlar: #[cite: 1]
    MAX_W = 1100
    MAX_H = 1500

    CROP_MARGIN_X = 0.03
    CROP_MARGIN_Y = 0.01

    ORB_FEATURES = 5000
    ORB_SCALE = 1.2
    ORB_LEVELS = 8
    ORB_EDGE_THRESHOLD = 15
    ORB_FAST_THRESHOLD = 10
    MATCH_RATIO = 0.72
    MIN_GOOD_MATCH = 12
    RANSAC_REPROJ = 4.0
    MIN_INLIERS = 10
    MIN_INLIER_RATIO = 0.25

    DIFF_BINARY_THRESHOLD = 35
    MIN_DIFF_AREA_RATIO = 0.0005
    BRIGHTNESS_DELTA = 25
    EDGE_MATCH_TOLERANCE = 0.06
    SCORE_THRESHOLD = 0.05
    CHANGE_THRESHOLD = 0.02


# ===========================================================
# DOSYA OKUMA / YAZMA YARDIMCILARI
# ===========================================================
def safe_imread(path: str) -> Optional[np.ndarray]: #[cite: 1]
    """Türkçe karakterli / Windows yollarını da okuyabilen imread."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ===========================================================
# GORUNTU ISLEME VE HIZALAMA
# ===========================================================
def normalize_gray(img: np.ndarray) -> np.ndarray: #[cite: 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def resize_keep_ratio(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray: #[cite: 1]
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img.copy()
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def crop_content(img: np.ndarray, margin_x: float, margin_y: float) -> np.ndarray: #[cite: 1]
    h, w = img.shape[:2]
    x1 = int(w * margin_x)
    x2 = int(w * (1 - margin_x))
    y1 = int(h * margin_y)
    y2 = int(h * (1 - margin_y))
    return img[y1:y2, x1:x2]


def align_images(reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, bool, int]: #[cite: 1]
    cv2.setRNGSeed(42)

    ref = reference.copy()
    tar = target.copy()

    h, w = ref.shape[:2]
    tar = cv2.resize(tar, (w, h), interpolation=cv2.INTER_AREA)

    gray_ref = normalize_gray(ref)
    gray_tar = normalize_gray(tar)

    orb = cv2.ORB_create(
        nfeatures=Ayarlar.ORB_FEATURES,
        scaleFactor=Ayarlar.ORB_SCALE,
        nlevels=Ayarlar.ORB_LEVELS,
        edgeThreshold=Ayarlar.ORB_EDGE_THRESHOLD,
        fastThreshold=Ayarlar.ORB_FAST_THRESHOLD,
    )

    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_tar, None)

    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return tar, False, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des2, des1, k=2)

    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < Ayarlar.MATCH_RATIO * n.distance:
            good.append(m)

    if len(good) < Ayarlar.MIN_GOOD_MATCH:
        return tar, False, len(good)

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, Ayarlar.RANSAC_REPROJ)

    if matrix is None:
        return tar, False, len(good)

    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / max(len(good), 1)

    if inliers < Ayarlar.MIN_INLIERS or inlier_ratio < Ayarlar.MIN_INLIER_RATIO:
        return tar, False, inliers

    aligned = cv2.warpPerspective(
        tar, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    return aligned, True, inliers


# ===========================================================
# DINAMIK RAF (BOLGE) ALGILAMA
# ===========================================================
def detect_shelf_lines(img: np.ndarray) -> List[int]: #[cite: 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)
    h, w = gray.shape

    kernel_w = max(40, w // 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    projection = np.sum(horizontal > 0, axis=1).astype(np.float32)
    projection = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 15), 0).ravel()

    threshold = max(w * 0.20, float(np.percentile(projection, 85)))
    candidates = np.where(projection >= threshold)[0]

    groups = []
    if len(candidates):
        start = candidates[0]
        prev = candidates[0]
        for y in candidates[1:]:
            if y - prev <= max(5, h // 50):
                prev = y
            else:
                groups.append((start, prev))
                start = y
                prev = y
        groups.append((start, prev))

    centers = []
    for a, b in groups:
        y = int((a + b) / 2)
        strength = float(np.max(projection[a:b + 1]))
        centers.append((y, strength))

    centers.sort(key=lambda x: x[0])
    merged: List[List[float]] = []
    min_gap = max(40, h // 6)

    for y, strength in centers:
        if not merged or y - merged[-1][0] > min_gap:
            merged.append([y, strength])
        else:
            if strength > merged[-1][1]:
                merged[-1] = [y, strength]

    lines = [int(y) for y, strength in merged if int(h * 0.05) < y < int(h * 0.95)]
    return lines


def build_shelves(img: np.ndarray, lines: List[int]) -> List[Tuple[int, int]]: #[cite: 1]
    h, w = img.shape[:2]
    valid = sorted(set([0] + lines + [h - 1]))

    cleaned = [valid[0]]
    for y in valid[1:]:
        if y - cleaned[-1] >= max(50, h // 8):
            cleaned.append(y)

    shelves = []
    for i in range(len(cleaned) - 1):
        y1 = cleaned[i]
        y2 = cleaned[i + 1]
        if y2 - y1 < max(40, h // 10):
            continue
        pad = max(2, int((y2 - y1) * 0.01))
        a = min(y1 + pad, y2 - 2)
        b = max(y2 - pad, a + 2)
        shelves.append((a, b))

    if not shelves:
        shelves.append((0, h - 1))

    return shelves


# ===========================================================
# BOLGE VE AKILLI ACIKLAMALI FARK TESPITI
# ===========================================================
@dataclass
class DiffBox: #[cite: 1]
    x: int
    y: int
    w: int
    h: int
    desc: str

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class ShelfAnalysis: #[cite: 1]
    y1: int
    y2: int
    change: float = 0.0
    edge_change: float = 0.0
    score: float = 0.0
    different: bool = False
    diff_boxes: List[DiffBox] = field(default_factory=list)


def analyze_shelf(ref_gray: np.ndarray, field_gray: np.ndarray, y1: int, y2: int) -> ShelfAnalysis: #[cite: 1]
    h, w = ref_gray.shape[:2]
    x1 = int(w * 0.03)
    x2 = int(w * 0.97)

    a = ref_gray[y1:y2, x1:x2]
    b = field_gray[y1:y2, x1:x2]

    if a.size == 0 or b.size == 0:
        return ShelfAnalysis(y1=y1, y2=y2)

    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    a_norm = cv2.normalize(a_blur, None, 0, 255, cv2.NORM_MINMAX)
    b_norm = cv2.normalize(b_blur, None, 0, 255, cv2.NORM_MINMAX)

    diff = cv2.absdiff(a_norm, b_norm)
    _, mask = cv2.threshold(diff, Ayarlar.DIFF_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(20, int(mask.size * Ayarlar.MIN_DIFF_AREA_RATIO))

    diff_boxes: List[DiffBox] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        bx = stats[i, cv2.CC_STAT_LEFT] + x1
        by = stats[i, cv2.CC_STAT_TOP] + y1
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        sub_a = a_norm[by - y1: by - y1 + bh, bx - x1: bx - x1 + bw]
        sub_b = b_norm[by - y1: by - y1 + bh, bx - x1: bx - x1 + bw]

        mean_a = float(np.mean(sub_a)) if sub_a.size > 0 else 0.0
        mean_b = float(np.mean(sub_b)) if sub_b.size > 0 else 0.0
        brightness_delta = abs(mean_b - mean_a)

        edge_a_box = cv2.Canny(sub_a, 50, 140)
        edge_b_box = cv2.Canny(sub_b, 50, 140)
        edge_mismatch = float(np.mean(cv2.absdiff(edge_a_box, edge_b_box) > 0)) if sub_a.size > 0 else 0.0

        is_pure_lighting = (
            brightness_delta > Ayarlar.BRIGHTNESS_DELTA
            and edge_mismatch < Ayarlar.EDGE_MATCH_TOLERANCE
        )
        if is_pure_lighting:
            continue

        desc = "Urun / Icerik Degisimi"
        diff_boxes.append(DiffBox(bx, by, bw, bh, desc))

    change = float(np.mean(mask > 0))
    edges_a = cv2.Canny(a_norm, 50, 140)
    edges_b = cv2.Canny(b_norm, 50, 140)
    edge_diff = cv2.absdiff(edges_a, edges_b)
    edge_change = float(np.mean(edge_diff > 0))

    score = (change * 0.6) + (edge_change * 0.4)
    different = (score > Ayarlar.SCORE_THRESHOLD and change > Ayarlar.CHANGE_THRESHOLD) or len(diff_boxes) > 0

    return ShelfAnalysis(
        y1=y1, y2=y2, change=change, edge_change=edge_change,
        score=score, different=different, diff_boxes=diff_boxes,
    )


# ===========================================================
# ANALIZ SONUCU VERI YAPISI
# ===========================================================
@dataclass
class AnalysisResult: #[cite: 1]
    result_image: np.ndarray
    total_shelves: int
    total_diffs: int
    avg_score: float
    aligned_ok: bool
    inliers: int
    report_text: str
    table_rows: List[dict]


def run_analysis(orj_path: str, saha_path: str) -> AnalysisResult: #[cite: 1]
    ref = safe_imread(orj_path)
    field_img = safe_imread(saha_path)

    if ref is None:
        raise ValueError(f"Orijinal fotograf okunamadi:\n{orj_path}")
    if field_img is None:
        raise ValueError(f"Saha fotografi okunamadi:\n{saha_path}")

    ref = resize_keep_ratio(ref, Ayarlar.MAX_W, Ayarlar.MAX_H)
    field_img = resize_keep_ratio(field_img, Ayarlar.MAX_W, Ayarlar.MAX_H)

    h, w = ref.shape[:2]
    field_img = cv2.resize(field_img, (w, h), interpolation=cv2.INTER_AREA)

    aligned, aligned_ok, inliers = align_images(ref, field_img)

    ref_work = crop_content(ref, Ayarlar.CROP_MARGIN_X, Ayarlar.CROP_MARGIN_Y)
    aligned_work = crop_content(aligned, Ayarlar.CROP_MARGIN_X, Ayarlar.CROP_MARGIN_Y)

    hh, ww = ref_work.shape[:2]
    aligned_work = cv2.resize(aligned_work, (ww, hh), interpolation=cv2.INTER_LINEAR)

    gray_ref = normalize_gray(ref_work)
    gray_saha = normalize_gray(aligned_work)

    lines = detect_shelf_lines(ref_work)
    shelf_ranges = build_shelves(ref_work, lines)

    result_img = aligned_work.copy()
    table_rows: List[dict] = []
    all_diff_items: List[str] = []
    diff_id = 1
    total_diffs = 0
    shelf_analyses: List[ShelfAnalysis] = []

    box_x1 = int(ww * 0.03)
    box_x2 = int(ww * 0.97)

    for idx, (y1, y2) in enumerate(shelf_ranges, start=1):
        analysis = analyze_shelf(gray_ref, gray_saha, y1, y2)
        shelf_analyses.append(analysis)

        if analysis.different:
            cv2.rectangle(result_img, (box_x1, y1), (box_x2, y2), (0, 0, 255), 2)

            for box in analysis.diff_boxes:
                cv2.rectangle(result_img, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 0, 255), 2)
                cv2.putText(
                    result_img, f"#{diff_id}", (box.x, max(box.y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
                )
                table_rows.append({
                    "Fark No": f"Fark #{diff_id}",
                    "Bölge": f"Bölge {idx}",
                    "Alan": f"{box.area} px",
                    "Akıllı Açıklama": box.desc
                })
                all_diff_items.append(
                    f"Fark #{diff_id} -> Bolge {idx} | Aciklama: {box.desc} (Alan: {box.area} px)"
                )
                diff_id += 1
                total_diffs += 1
        else:
            cv2.rectangle(result_img, (box_x1, y1), (box_x2, y2), (0, 180, 0), 1)

    if total_diffs == 0:
        table_rows.append({
            "Fark No": "-",
            "Bölge": "Tümü",
            "Alan": "0 px",
            "Akıllı Açıklama": "Tam Uyumlu"
        })

    avg_score = float(np.mean([s.score for s in shelf_analyses])) if shelf_analyses else 0.0

    report_lines = [
        "=== DENETIM RAPORU - TESPIT EDILEN FARKLAR ===",
        f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
        "",
    ]
    report_lines.extend(all_diff_items if all_diff_items else ["Hicbir fark bulunamadi, sistem tam uyumlu."])

    return AnalysisResult(
        result_image=result_img,
        total_shelves=len(shelf_ranges),
        total_diffs=total_diffs,
        avg_score=avg_score,
        aligned_ok=aligned_ok,
        inliers=inliers,
        report_text="\n".join(report_lines),
        table_rows=table_rows,
    )


# ===========================================================
# STREAMLIT ARAYÜZÜ
# ===========================================================
st.set_page_config(page_title="Sigara Standı Akıllı Denetim Paneli", layout="wide")

st.title("Sigara Standı Akıllı Denetim Paneli")
st.markdown("Akıllı Açıklamalı Numaralandırılmış Fark Analizi v4.0")
st.markdown("---")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    uploaded_orj = st.file_uploader("Orijinal Fotoğrafı Seçin", type=["jpg", "jpeg", "png", "webp"])

with col_upload2:
    uploaded_saha = st.file_uploader("Sahadan Gelen Fotoğrafı Seçin", type=["jpg", "jpeg", "png", "webp"])

if uploaded_orj and uploaded_saha:
    # Geçici dosyalar oluşturarak yol uyumluluğunu sağla
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f_orj:
        f_orj.write(uploaded_orj.getvalue())
        orj_path = f_orj.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f_saha:
        f_saha.write(uploaded_saha.getvalue())
        saha_path = f_saha.name

    if st.button("ANALİZ ET VE GÖRSELİ GÖSTER", type="primary", use_container_width=True):
        if orj_path == saha_path:
            st.warning("Orijinal ve saha fotoğrafı aynı dosya olamaz.")
        else:
            with st.spinner("Görüntüler hizalanıyor ve akıllı farklar tespit ediliyor..."):
                try:
                    result = run_analysis(orj_path, saha_path)
                    
                    st.success(f"Analiz tamamlandı. Toplam {result.total_diffs} fark tespit edildi.")
                    
                    # Özet Bilgiler
                    col_info1, col_info2, col_info3 = st.columns(3)
                    col_info1.metric("Algılanan Bölge", result.total_shelves)
                    col_info2.metric("Toplam Fark", result.total_diffs)
                    col_info3.metric("Ortalama Skor", f"%{result.avg_score * 100:.1f}")

                    if not result.aligned_ok:
                        st.warning("[!] Görüntü hizalama başarısız oldu, sonuçlar güvenilir olmayabilir.")

                    # Sonuç Görseli ve Tablo Düzeni
                    col_res1, col_res2 = st.columns([1.3, 1])

                    with col_res1:
                        st.subheader("İşaretli Sonuç Görseli")
                        result_rgb = cv2.cvtColor(result.result_image, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)

                        # Görsel İndirme Butonu
                        is_success, encoded_img = cv2.imencode(".jpg", result.result_image)
                        if is_success:
                            st.download_button(
                                label="Sonuç Görselini İndir",
                                data=encoded_img.tobytes(),
                                file_name=f"analiz_sonucu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg"
                            )

                    with col_res2:
                        st.subheader("Tespit Edilen Farklar")
                        st.dataframe(result.table_rows, use_container_width=True)

                        st.subheader("Denetim Raporu")
                        st.text_area("Rapor İçeriği", result.report_text, height=150)
                        
                        st.download_button(
                            label="Raporu Metin Olarak İndir",
                            data=result.report_text,
                            file_name=f"denetim_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

                except Exception as exc:
                    st.error(f"Analiz sırasında bir hata oluştu:\n{exc}")

                finally:
                    # Geçici dosyaları temizle
                    if os.path.exists(orj_path):
                        os.unlink(orj_path)
                    if os.path.exists(saha_path):
                        os.unlink(saha_path)
else:
    st.info("Lütfen analiz için her iki fotoğrafı da yükleyin.")
