# =========================================================
# SONUÇ EKRANI (DÜZELTİLMİŞ)
# =========================================================
if (
    st.session_state.result_img is not None
    and st.session_state.summary
):
    summary = st.session_state.summary
    
    # Doğru toplam hesaplaması (Eksik paketler + Etiket farkları)
    paket_sayisi = int(summary.get("paket_eksigi", 0))
    etiket_sayisi = int(summary.get("etiket_degisikligi", 0))
    toplam_degisiklik = paket_sayisi + etiket_sayisi

    st.subheader("3. Analiz Sonucu")

    st.metric(
        "🔴 TOPLAM TESPİT EDİLEN DEĞİŞİKLİK",
        toplam_degisiklik,
    )

    st.image(
        st.session_state.result_img,
        channels="BGR",
        use_container_width=True,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        st.session_state.result_img,
    )
    if ok:
        st.download_button(
            "📥 İşaretli Denetim Görselini İndir",
            data=encoded.tobytes(),
            file_name=(
                f"{(dealer_name or 'planogram').replace(' ', '_')}"
                "_denetim.jpg"
            ),
            mime="image/jpeg",
            use_container_width=True,
        )

else:
    st.info(
        "Analiz için referans ve saha fotoğrafını yükleyin, "
        "ardından 'KONTROLE BAŞLA' düğmesine basın."
    )
