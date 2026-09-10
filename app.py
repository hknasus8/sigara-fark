# -*- coding: utf-8 -*-
"""
SIGARA STANDI AKILLI DENETIM SISTEMI - v4.0
---------------------------------------------------------
Onceki surumden (v3.3) duzeltilen hatalar:
  1) analyze_shelf() her raf icin iki kez cagriliyordu (cizim + ortalama
     skor hesabi) -> tek seferde hesaplanip sonuc yeniden kullaniliyor.
  2) Treeview sutun genislik sozlugunde "durum" yerine yanlislikla
     "decor" anahtari kullanilmisti -> sutun genisligi duzeltildi.
  3) Gorsel hizalama basarisiz oldugunda kullaniciya bildirilmiyordu
     -> hizalama durumu hem arayuzde hem raporda gosteriliyor.
  4) Analiz islemi ana thread'de calisip arayuzu kilitliyordu
     -> arka plan thread'i + ilerleme cubugu eklendi.
  5) Rapor basliginda "AKILLİ" / "AKILLI" tutarsizligi vardi
     -> Turkce buyuk harf kullanimi tutarli hale getirildi.
  6) Beklenmeyen hatalarda program cokebiliyordu
     -> analiz calisan thread'de try/except ile korunuyor.
  7) Sonuc gorseli diske kaydedilemiyordu
     -> "Gorseli Kaydet" butonu eklendi.
---------------------------------------------------------
"""

import os
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


# ===========================================================
# AYARLAR (tum esik degerleri tek yerden yonetiliyor)
# ===========================================================
class Ayarlar:
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
def safe_imread(path: str) -> Optional[np.ndarray]:
    """Turkce karakterli / Windows yollarini da okuyabilen imread."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_imwrite(path: str, image: np.ndarray) -> bool:
    """Turkce karakterli yollar icin guvenli goruntu kaydi."""
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, encoded = cv2.imencode(ext, image)
        if ok:
            encoded.tofile(path)
            return True
    except Exception:
        pass
    return False


# ===========================================================
# GORUNTU ISLEME VE HIZALAMA
# ===========================================================
def normalize_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def resize_keep_ratio(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img.copy()
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def crop_content(img: np.ndarray, margin_x: float, margin_y: float) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int(w * margin_x)
    x2 = int(w * (1 - margin_x))
    y1 = int(h * margin_y)
    y2 = int(h * (1 - margin_y))
    return img[y1:y2, x1:x2]


def align_images(reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    """ORB + homografi ile hedef gorseli referansa hizalar.
    Donus: (hizalanmis_goruntu, basarili_mi, inlier_sayisi)
    """
    # DUZELTME: cv2.findHomography(..., cv2.RANSAC, ...) icsel olarak
    # rastgele nokta alt kumeleri deneyerek calisir. Sabit bir tohum
    # verilmezse, AYNI iki fotografla bile her calistirmada FARKLI bir
    # hizalama (ve dolayisiyla farkli fark sonuclari) uretebiliyordu.
    # Sabit seed ile sonuc artik deterministik: ayni girdi -> ayni cikti.
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
def detect_shelf_lines(img: np.ndarray) -> List[int]:
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


def build_shelves(img: np.ndarray, lines: List[int]) -> List[Tuple[int, int]]:
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
class DiffBox:
    x: int
    y: int
    w: int
    h: int
    desc: str

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class ShelfAnalysis:
    y1: int
    y2: int
    change: float = 0.0
    edge_change: float = 0.0
    score: float = 0.0
    different: bool = False
    diff_boxes: List[DiffBox] = field(default_factory=list)


def analyze_shelf(ref_gray: np.ndarray, field_gray: np.ndarray, y1: int, y2: int) -> ShelfAnalysis:
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

        # DUZELTME (v4.1): oncekı surumde sadece ortalama parlaklik farkina
        # bakilarak filtreleme yapiliyordu. Bu, bos bir alana urun eklenmesi
        # / kaldirilmasi gibi GERCEK icerik degisikliklerini de yanlislikla
        # "isik/golge" sanip siliyordu (cunku bos alan <-> urun gecisi de
        # ortalama parlakligi buyuk olcude degistirir).
        #
        # Simdi hem parlaklik farkina HEM de kutu icindeki kenar (desen)
        # yapisina birlikte bakiliyor:
        #   - Parlaklik degismis AMA kenar deseni (aynı urunun hatlari,
        #     yazilari) hala buyuk olcude ortusuyorsa -> gercekten sadece
        #     isik/golgedir, atla.
        #   - Parlaklik degismis VE kenar deseni de tamamen farkliysa
        #     (bos yer -> urun, ya da farkli bir urun) -> gercek icerik
        #     degisimidir, listede tut.
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
            continue  # sadece isik/golge -> icerik degisimi degil, atla

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
# ANALIZ SONUCU VE UC-KATMAN AYRIMI (UI'DAN BAGIMSIZ HESAPLAMA)
# ===========================================================
@dataclass
class AnalysisResult:
    result_image: np.ndarray
    total_shelves: int
    total_diffs: int
    avg_score: float
    aligned_ok: bool
    inliers: int
    report_text: str
    table_rows: List[Tuple[str, str, str, str, str]]  # no, bolge, boyut, aciklama, tag


def run_analysis(orj_path: str, saha_path: str) -> AnalysisResult:
    """Iki fotografi karsilastirip AnalysisResult uretir.
    Tum agir islem burada, arayuzden bagimsiz olarak yapilir; boylece
    arka plan thread'inde calistirilabilir ve tek yerden test edilebilir.
    """
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
    table_rows: List[Tuple[str, str, str, str, str]] = []
    all_diff_items: List[str] = []
    diff_id = 1
    total_diffs = 0
    shelf_analyses: List[ShelfAnalysis] = []

    box_x1 = int(ww * 0.03)
    box_x2 = int(ww * 0.97)

    for idx, (y1, y2) in enumerate(shelf_ranges, start=1):
        # DUZELTME: analyze_shelf artik her raf icin sadece BIR kez
        # cagriliyor; sonuc hem cizim hem ortalama skor icin kullaniliyor.
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
                table_rows.append((f"Fark #{diff_id}", f"Bolge {idx}", f"{box.area} px", box.desc, "fark"))
                all_diff_items.append(
                    f"Fark #{diff_id} -> Bolge {idx} | Aciklama: {box.desc} (Alan: {box.area} px)"
                )
                diff_id += 1
                total_diffs += 1
        else:
            cv2.rectangle(result_img, (box_x1, y1), (box_x2, y2), (0, 180, 0), 1)

    if total_diffs == 0:
        table_rows.append(("-", "Tumu", "0 px", "Tam Uyumlu", "normal"))

    avg_score = float(np.mean([s.score for s in shelf_analyses])) if shelf_analyses else 0.0

    # DUZELTME: rapor artik yalnizca tespit edilen farklari listeliyor;
    # tarih/dosya/hizalama/bolge gibi ozet bilgiler rapor metninden
    # cikarildi (bu bilgiler zaten arayuzdeki "Ozet" panelinde gosteriliyor).
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
# ANA ARAYUZ
# ===========================================================
class SigaraFarkDashboard:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sigara Standi Akilli Denetim Paneli v4.0")
        self.root.geometry("1550x950")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#15191f")

        self.orj_path = ""
        self.saha_path = ""
        self.current_result: Optional[AnalysisResult] = None
        self.current_cv_image: Optional[np.ndarray] = None
        self.zoom_factor = 1.0
        self.is_busy = False

        self.build_ui()

    # -------------------------------------------------------
    # ARAYUZ KURULUMU
    # -------------------------------------------------------
    def build_ui(self):
        top = tk.Frame(self.root, bg="#20262e", height=65)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="AKILLI DENETIM SISTEMI",
            fg="white", bg="#20262e", font=("Arial", 17, "bold")
        ).pack(side=tk.LEFT, padx=18)

        tk.Label(
            top, text="Akilli Aciklamali Numaralandirilmis Fark Analizi",
            fg="#b9c1cc", bg="#20262e", font=("Arial", 10)
        ).pack(side=tk.LEFT)

        main = tk.Frame(self.root, bg="#15191f")
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = tk.Frame(main, bg="#20262e", width=820)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        right = tk.Frame(main, bg="#20262e", width=520)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(6, 0))

        # --- Dosya secimi ---
        file_frame = tk.LabelFrame(left, text="Fotograf Secimi", fg="white", bg="#20262e", font=("Arial", 10, "bold"))
        file_frame.pack(fill=tk.X, padx=12, pady=10)

        f_row1 = tk.Frame(file_frame, bg="#20262e")
        f_row1.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(f_row1, text="Orjinal Sec", command=self.select_orj, bg="#3a4350", fg="white", relief=tk.FLAT, width=15).pack(side=tk.LEFT)
        self.lbl_orj = tk.Label(f_row1, text="Secilmedi", fg="#8fa4b8", bg="#20262e", anchor="w")
        self.lbl_orj.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        f_row2 = tk.Frame(file_frame, bg="#20262e")
        f_row2.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(f_row2, text="Saha Foto Sec", command=self.select_saha, bg="#3a4350", fg="white", relief=tk.FLAT, width=15).pack(side=tk.LEFT)
        self.lbl_saha = tk.Label(f_row2, text="Secilmedi", fg="#8fa4b8", bg="#20262e", anchor="w")
        self.lbl_saha.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        f_row3 = tk.Frame(file_frame, bg="#20262e")
        f_row3.pack(fill=tk.X, padx=8, pady=6)
        self.btn_analiz = tk.Button(
            f_row3, text="ANALIZ ET VE GORSELI GOSTER", command=self.analiz_et_baslat,
            bg="#168a45", fg="white", relief=tk.FLAT, font=("Arial", 10, "bold"), padx=12, pady=6
        )
        self.btn_analiz.pack(fill=tk.X)

        self.progress = ttk.Progressbar(file_frame, mode="indeterminate")
        # Sadece analiz sirasinda pack edilir (build_ui'da gizli tutulur).

        # --- Zoom kontrolleri ---
        zoom_frame = tk.Frame(left, bg="#20262e")
        zoom_frame.pack(fill=tk.X, padx=12, pady=(0, 6))

        tk.Label(zoom_frame, text="Buyutec:", fg="#b9c1cc", bg="#20262e", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(zoom_frame, text=" + Buyut ", command=self.zoom_in, bg="#2c3540", fg="white", relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text=" - Kucult ", command=self.zoom_out, bg="#2c3540", fg="white", relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text=" Sifirla ", command=self.zoom_reset, bg="#2c3540", fg="white", relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        self.lbl_zoom_info = tk.Label(zoom_frame, text="Oran: %100", fg="#8fa4b8", bg="#20262e", font=("Arial", 9))
        self.lbl_zoom_info.pack(side=tk.LEFT, padx=10)

        tk.Button(
            zoom_frame, text="Gorseli Kaydet", command=self.save_image,
            bg="#2c3540", fg="white", relief=tk.FLAT, padx=8, pady=4
        ).pack(side=tk.RIGHT, padx=2)

        # --- Goruntu alani ---
        image_box = tk.LabelFrame(left, text="Analiz Sonucu Gorseli (Isaretli Bolgeler & Numaralar)", fg="white", bg="#20262e", font=("Arial", 10, "bold"))
        image_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        canvas_container = tk.Frame(image_box, bg="#080a0d")
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg="#080a0d", highlightthickness=0)
        hbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_durum = tk.Label(
            left, text="Durum: Lutfen fotograflari secin.", fg="#dfe6ee", bg="#20262e", anchor="w", font=("Arial", 10, "bold")
        )
        self.lbl_durum.pack(fill=tk.X, padx=12, pady=(0, 12))

        # --- Sag panel: ozet ---
        summary = tk.LabelFrame(right, text="Ozet", fg="white", bg="#20262e", font=("Arial", 10, "bold"))
        summary.pack(fill=tk.X, padx=12, pady=12)

        self.lbl_ozet = tk.Label(
            summary, text="Henuz analiz yapilmadi.", justify=tk.LEFT, anchor="w",
            fg="#dfe6ee", bg="#20262e", font=("Arial", 10), padx=10, pady=10
        )
        self.lbl_ozet.pack(fill=tk.X)

        # --- Sag panel: tablo ---
        table_frame = tk.LabelFrame(right, text="Tespit Edilen Numarali Farklar ve Akilli Aciklamalar", fg="white", bg="#20262e", font=("Arial", 10, "bold"))
        table_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        columns = ("no", "bolge", "boyut", "durum")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)

        headings = {"no": "Fark No", "bolge": "Bolge", "boyut": "Alan", "durum": "Akilli Aciklama"}
        # DUZELTME: sozluk anahtarlari artik gercek sutun adlariyla eslesiyor
        # (eskiden "durum" yerine yanlislikla "decor" kullanilmisti).
        widths = {"no": 70, "bolge": 70, "boyut": 90, "durum": 170}

        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="center")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.tag_configure("fark", background="#ffd7d7")
        self.tree.tag_configure("normal", background="#dff5e5")

        # --- Sag panel: rapor ---
        report_frame = tk.LabelFrame(right, text="Denetim Raporu", fg="white", bg="#20262e", font=("Arial", 10, "bold"))
        report_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.txt_rapor = tk.Text(
            report_frame, wrap=tk.WORD, bg="#0e1217", fg="#e5ebf2",
            insertbackground="white", font=("Consolas", 9)
        )
        self.txt_rapor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        tk.Button(
            report_frame, text="Raporu Kaydet", command=self.save_report,
            bg="#3a4350", fg="white", relief=tk.FLAT, padx=10, pady=7
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

    # -------------------------------------------------------
    # DOSYA SECIMI
    # -------------------------------------------------------
    def select_orj(self):
        path = filedialog.askopenfilename(
            title="Orijinal fotografi secin",
            filetypes=[("Resim Dosyalari", "*.jpg *.jpeg *.png *.webp")],
        )
        if path:
            self.orj_path = path
            self.lbl_orj.config(text=os.path.basename(path), fg="#71fc79")

    def select_saha(self):
        path = filedialog.askopenfilename(
            title="Sahadan gelen fotografi secin",
            filetypes=[("Resim Dosyalari", "*.jpg *.jpeg *.png *.webp")],
        )
        if path:
            self.saha_path = path
            self.lbl_saha.config(text=os.path.basename(path), fg="#71fc79")

    # -------------------------------------------------------
    # GORUNTU GOSTERIMI / ZOOM
    # -------------------------------------------------------
    def show_image(self):
        if self.current_cv_image is None:
            return

        img = self.current_cv_image
        h, w = img.shape[:2]

        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)

        resized = cv2.resize(img, (max(50, new_w), max(50, new_h)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self.tk_img = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.lbl_zoom_info.config(text=f"Oran: %{int(self.zoom_factor * 100)}")

    def zoom_in(self):
        if self.current_cv_image is not None:
            self.zoom_factor = min(5.0, self.zoom_factor + 0.25)
            self.show_image()

    def zoom_out(self):
        if self.current_cv_image is not None:
            self.zoom_factor = max(0.5, self.zoom_factor - 0.25)
            self.show_image()

    def zoom_reset(self):
        if self.current_cv_image is not None:
            self.zoom_factor = 1.0
            self.show_image()

    # -------------------------------------------------------
    # ANALIZ (ARKA PLAN THREAD'I ILE)
    # -------------------------------------------------------
    def analiz_et_baslat(self):
        if self.is_busy:
            return

        if not self.orj_path or not self.saha_path:
            messagebox.showwarning("Uyari", "Lutfen hem orijinal hem de saha fotografini secin.")
            return

        if self.orj_path == self.saha_path:
            messagebox.showwarning("Uyari", "Orijinal ve saha fotografi ayni dosya olamaz.")
            return

        self.is_busy = True
        self.btn_analiz.config(state=tk.DISABLED, text="ANALIZ EDILIYOR...")
        self.progress.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.progress.start(12)
        self.lbl_durum.configure(text="Durum: Goruntuler hizalaniyor ve akilli farklar tespit ediliyor...")

        # DUZELTME: agir islem artik ayri bir thread'de calisiyor,
        # boylece arayuz analiz surerken kilitlenmiyor.
        orj_path, saha_path = self.orj_path, self.saha_path
        thread = threading.Thread(target=self._analiz_worker, args=(orj_path, saha_path), daemon=True)
        thread.start()

    def _analiz_worker(self, orj_path: str, saha_path: str):
        try:
            result = run_analysis(orj_path, saha_path)
            self.root.after(0, self._analiz_bitti, result, None)
        except Exception as exc:
            traceback.print_exc()
            self.root.after(0, self._analiz_bitti, None, str(exc))

    def _analiz_bitti(self, result: Optional[AnalysisResult], error: Optional[str]):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_analiz.config(state=tk.NORMAL, text="ANALIZ ET VE GORSELI GOSTER")
        self.is_busy = False

        if error is not None:
            messagebox.showerror("Hata", f"Analiz sirasinda bir hata olustu:\n\n{error}")
            self.lbl_durum.configure(text="Durum: Analiz basarisiz oldu.")
            return

        assert result is not None

        for item in self.tree.get_children():
            self.tree.delete(item)
        for no, bolge, boyut, aciklama, tag in result.table_rows:
            self.tree.insert("", tk.END, values=(no, bolge, boyut, aciklama), tags=(tag,))

        self.txt_rapor.delete("1.0", tk.END)
        self.txt_rapor.insert(tk.END, result.report_text)

        self.current_cv_image = result.result_image
        self.zoom_factor = 1.0
        self.show_image()

        hizalama_uyarisi = "" if result.aligned_ok else "\n[!] Hizalama basarisiz oldu, sonuclar guvenilir olmayabilir."
        self.lbl_ozet.configure(
            text=(
                f"Algilanan Bolge: {result.total_shelves}\n"
                f"Toplam Fark: {result.total_diffs}\n"
                f"Ort. Skor: %{result.avg_score * 100:.1f}"
                f"{hizalama_uyarisi}"
            )
        )

        durum_metni = f"Durum: Analiz tamamlandi. {result.total_diffs} fark akilli aciklamalarla listelendi."
        if not result.aligned_ok:
            durum_metni += " (Uyari: goruntu hizalama basarisiz oldu)"
        self.lbl_durum.configure(text=durum_metni)

        self.current_result = result

    # -------------------------------------------------------
    # KAYIT ISLEMLERI
    # -------------------------------------------------------
    def save_report(self):
        if not self.current_result:
            messagebox.showinfo("Bilgi", "Once analiz yapin.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Metin dosyasi", "*.txt")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.current_result.report_text)
                messagebox.showinfo("Basarili", "Rapor kaydedildi.")
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    def save_image(self):
        if self.current_cv_image is None:
            messagebox.showinfo("Bilgi", "Once analiz yapin.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if path:
            if safe_imwrite(path, self.current_cv_image):
                messagebox.showinfo("Basarili", "Gorsel kaydedildi.")
            else:
                messagebox.showerror("Hata", "Gorsel kaydedilemedi.")


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    try:
        app = SigaraFarkDashboard(root)
        root.mainloop()
    except Exception:
        traceback.print_exc()
        try:
            messagebox.showerror("Beklenmeyen Hata", "Uygulama baslatilirken bir hata olustu.\nDetaylar konsolda.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
