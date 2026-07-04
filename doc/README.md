# PCA Image Compressor 
Project ini bertujuan untuk melakukan kompresi gambar menggunakan algoritma Principal Component Analysis (PCA). Dengan mereduksi dimensi data gambar melalui dekomposisi nilai eigen (singular value decomposition), sistem ini mampu mengkompresi gambar dengan tetap mempertahankan karakteristik visual dan warna aslinya.

## Persyaratan 
Menginstal Python (versi 3.8 atau lebih baru). 
Library yang dibutuhkan:
- streamlit
- opencv-python
- numpy

## Installasi
- Ketik: `pip install streamlit opencv-python numpy`
- Cara membuka project: `streamlit run src/imagecompres.py`

## Cara Menjalankan
   - Di sidebar, masukkan foto yang ingin dikompresi.
   - Pilih tingkatan kompresi yang diinginkan, semakin kecil maka gambar yang dihasilkan akan semakin buram.
   - Tunggu beberapa saat hingga proses selesai
   - Hasil kompresi akan muncul di layar utama bersamaan dengan gambar input asli.
   - Periksa metrik evaluasi seperti waktu algoritma dalam menjalankan kompresi dan persentase hasil kompresi gambar.
   - Klik tombol "DOWNLOAD HASIL KOMPRESI" jika ingin menyimpan hasil kompresi.