import pandas as pd
from google_play_scraper import reviews_all, Sort

def main():
    print("Proses akuisisi data ulasan aplikasi JAKI dari Google Play Store...")
    # Menggunakan library google-play-scraper untuk mengambil data berdasarkan ID aplikasi
    scrapreview = reviews_all(
        'id.go.jakarta.smartcity.jaki',
        lang='id',
        country='id',
        sort=Sort.MOST_RELEVANT,
        count=12000
    )

    # Transformasi hasil scraping menjadi struktur DataFrame
    app_reviews_df = pd.DataFrame(scrapreview)
    print(f"Total data ulasan yang berhasil dikumpulkan: {len(app_reviews_df)}")

    # Simpan seluruh dataset mentah hasil scraping ke CSV
    output_filename = 'ulasan_aplikasi_jaki_raw.csv'
    app_reviews_df.to_csv(output_filename, index=False)

    jumlah_ulasan, jumlah_kolom = app_reviews_df.shape
    print(f"Dataset berhasil disimpan ke '{output_filename}'! Total: {jumlah_ulasan} baris, {jumlah_kolom} kolom.")

if __name__ == '__main__':
    main()
