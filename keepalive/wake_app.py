# -*- coding: utf-8 -*-
import os
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://sigarafark.streamlit.app/",
)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Açılıyor: {APP_URL}")
        page.goto(APP_URL, wait_until="networkidle", timeout=60000)

        # Streamlit uyku ekranındaki 'Yes, get this app back up!' butonunu bulmak için seçiciler
        woke_up = False
        selectors = [
            "button:has-text('get this app back up')",
            "button:has-text('Yes')",
            "text=get this app back up",
            "text=Wake it back up"
        ]

        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0 and btn.is_visible():
                    print(f"Uyku düğmesi bulundu ({selector}), tıklanıyor...")
                    btn.click(timeout=5000)
                    woke_up = True
                    break
            except Exception as e:
                print(f"Seçici denenirken hata: {e}")

        if woke_up:
            print("Uyandırma isteği gönderildi, uygulamanın yüklenmesi bekleniyor...")
            # Uygulamanın tam olarak ayağa kalkması için biraz daha uzun bekleyelim
            page.wait_for_timeout(20000)
        else:
            print("Uyku ekranı algılanmadı, uygulama zaten aktif olabilir.")

        content = page.content().lower()
        browser.close()

        if "this app has gone to sleep" in content:
            print("UYARI: Uygulama hâlâ uyku modunda.")
            sys.exit(1)

        print("Tamam: Uygulama aktif durumda.")

if __name__ == "__main__":
    for attempt in range(1, 4):
        try:
            main()
            break
        except Exception as exc:
            print(f"Deneme {attempt}/3 başarısız: {exc}")
            if attempt == 3:
                sys.exit(1)
            time.sleep(10)
