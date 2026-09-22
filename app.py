def find_label_bands(gray, top_y, bottom_y):
    roi = gray[top_y:bottom_y, :]
    if roi.size == 0:
        return []

    # Işık değişimlerine karşı daha esnek bir eşikleme (Otsu veya dinamik yaklaşım)
    # 150 sabit sınırını düşürerek veya adaptif eşik kullanarak daha çok alan yakalayalım
    _, bright = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    bright01 = (bright > 0).astype(np.uint8)

    seg_counts = np.array(
        [_count_wide_segments(bright01[y]) for y in range(bright01.shape[0])],
        dtype=np.float32,
    )
    if seg_counts.size == 0:
        return []

    smooth = np.convolve(seg_counts, np.ones(5) / 5.0, mode="same")
    
    # Kriteri esnetiyoruz: 6 yerine 3 veya daha düşük bir yoğunluk yeterli olsun
    is_label_row = smooth >= 3

    bands = []
    in_band = False
    start = 0
    for y, v in enumerate(is_label_row):
        if v and not in_band:
            start = y
            in_band = True
        elif not v and in_band:
            bands.append((start, y))
            in_band = False
    if in_band:
        bands.append((start, len(is_label_row)))

    # Etiket yükseklik toleransını genişletelim (10 ile 50 piksel arası)
    return [(b[0] + top_y, b[1] + top_y) for b in bands if 10 <= (b[1] - b[0]) <= 50]


def segment_label_cells(gray, band_top, band_bot):
    band = gray[band_top:band_bot, :]
    if band.size == 0:
        return []

    _, mask = cv2.threshold(band, 120, 255, cv2.THRESH_BINARY)
    mask01 = (mask > 0).astype(np.float32)
    col_frac = mask01.mean(axis=0)

    # Sütun doluluk oranını esnetelim (0.4 yerine 0.25)
    is_label_col = col_frac > 0.25
    x_ranges = []
    in_cell = False
    start = 0
    for x, v in enumerate(is_label_col):
        if v and not in_cell:
            start = x
            in_cell = True
        elif not v and in_cell:
            x_ranges.append((start, x))
            in_cell = False
    if in_cell:
        x_ranges.append((start, len(is_label_col)))

    cells = []
    for x1, x2 in x_ranges:
        bw = x2 - x1
        if bw < 10 or bw > 120:
            continue
        row_frac = mask01[:, x1:x2].mean(axis=1)
        
        # Satır doluluk oranını esnetelim (0.75 yerine 0.50)
        rows = np.where(row_frac > 0.50)[0]
        if rows.size == 0:
            continue
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        bh = y2 - y1
        if bh < 5:
            continue
        cells.append((x1, band_top + y1, bw, bh))
    return cells
