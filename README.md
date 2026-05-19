# Watermarking — Evaluasi Ketahanan terhadap Kompresi JPEG

Mata Kuliah: Sinyal & Multimedia

---

## Persiapan

### Install library yang dibutuhkan
Buka terminal / command prompt, jalankan:
```bash
pip install numpy pillow matplotlib scipy
```

### Struktur folder
```
project/
├── watermarking.py     ← script utama
├── foto_wajah.jpg      ← foto wajahmu (taruh di sini)
└── README.md
```

### Ganti nama foto
Buka `watermarking.py`, ubah baris ini sesuai nama file fotomu:
```python
FOTO_PATH = "foto_wajah.jpg"   # <-- ganti ini
```

### Cara menjalankan
```bash
python watermarking.py
```

---

## Penjelasan Per Step

### Step 1 — Cek File Foto
Script mengecek apakah file foto ada di folder yang sama. Kalau tidak ditemukan, program berhenti dan kasih tahu lokasi folder saat ini supaya kamu bisa naruh foto di tempat yang benar.

---

### Step 2 — Load Gambar sebagai Grayscale
Foto wajah dibuka lalu dikonversi ke mode **grayscale** (hitam-putih) menggunakan PIL. Hasilnya berupa array 2D NumPy di mana setiap elemen adalah nilai piksel 0–255.

```
Output: output_step2_gambar_asli.png
```

Kenapa grayscale? Karena watermarking LSB bekerja pada satu channel piksel. Foto berwarna (RGB) punya 3 channel, kalau dipakai semua jadi lebih kompleks — untuk praktikum ini cukup grayscale.

---

### Step 3 — Buat Watermark Biner Acak
Dibuat sebuah array 64×64 pixel berisi nilai **0 dan 1** secara acak menggunakan `np.random.seed(42)`. `seed(42)` dipakai supaya watermark yang sama selalu terbentuk setiap kali program dijalankan (reproducible).

```
Output: output_step3_watermark.png
```

Watermark ini nantinya yang akan disisipkan ke foto dan kemudian diekstrak kembali setelah kompresi untuk mengukur seberapa banyak yang hilang.

---

### Step 4 — Embed Watermark (Metode LSB)
**LSB = Least Significant Bit** — bit paling kanan dari setiap nilai piksel.

Cara kerjanya:
1. Setiap piksel punya nilai 8-bit, misalnya `10110110`
2. Bit terakhir (LSB) diganti dengan bit watermark
3. `10110110` → `10110111` (berubah 1 nilai dari 182 ke 183)

Perubahan sebesar 1 ini **tidak terlihat oleh mata manusia**, tapi bisa dibaca oleh program.

```python
img_wm = (img_array & 0b11111110) | watermark
# 0b11111110 = matikan bit terakhir, lalu OR dengan bit watermark
```

```
Output: output_step4_embed.png
         wajah_watermarked.png
```

---

### Step 5 — Visualisasi DCT & Quantization
Ini bagian penjelasan **bagaimana JPEG bekerja** secara internal, yang menjelaskan kenapa watermark bisa hilang saat kompresi.

Proses JPEG per blok 8×8 piksel:
1. **DCT (Discrete Cosine Transform)** — ubah piksel jadi koefisien frekuensi
2. **Quantization** — bagi koefisien dengan tabel quantization, lalu bulatkan
3. **Rekonstruksi** — kalikan balik, lalu IDCT

Quality Factor (QF) mengontrol tabel quantization:
- **QF rendah (10)** → nilai tabel besar → banyak koefisien jadi 0 → banyak info hilang → file kecil
- **QF tinggi (100)** → nilai tabel kecil → sedikit info hilang → file besar

LSB adalah frekuensi **sangat tinggi** (detail halus), sehingga quantization QF rendah langsung menghapusnya.

```
Output: output_step5_dct.png
```

---

### Step 6 — Kompresi JPEG Berbagai Quality Factor
Foto yang sudah diwatermark disimpan ulang sebagai JPEG dengan QF berbeda: `10, 20, 30, 50, 70, 90, 100`.

Semakin rendah QF:
- File makin kecil
- Kualitas gambar makin jelek
- Watermark makin rusak

```
Output: output_step6_kompresi.png
         output_step6_ukuran_file.png
         wajah_qf10.jpg ... wajah_qf100.jpg
```

---

### Step 7 — Ekstraksi Watermark & Perhitungan BER
Dari setiap file JPEG hasil kompresi, watermark diekstrak kembali dengan cara yang sama: ambil LSB setiap piksel di area 64×64.

**BER (Bit Error Rate)** dihitung sebagai:

```
BER = jumlah bit yang salah / total bit watermark
```

- BER = 0.00 → watermark sempurna, tidak ada yang rusak
- BER = 0.50 → watermark acak total, sama seperti tebak-tebakan
- BER < 0.10 → watermark masih bisa diekstrak (threshold praktis)

```
Output: output_step7_ekstraksi.png
```

---

### Step 8 — Grafik BER vs Quality Factor
Grafik garis yang menunjukkan hubungan antara QF dan BER. Garis merah putus-putus adalah threshold 0.1. Area merah menunjukkan zona di mana watermark sudah tidak bisa diekstrak.

**Kesimpulan yang diharapkan:**
- QF rendah (10–30) → BER tinggi → watermark **tidak bisa** diekstrak
- QF tinggi (70–100) → BER rendah → watermark **masih bisa** diekstrak
- Catat QF berapa yang jadi batas (BER mulai ≥ 0.1)

```
Output: output_step8_grafik_ber.png
```

---

## Semua File Output

| File | Isi |
|------|-----|
| `output_step2_gambar_asli.png` | Foto wajah grayscale |
| `output_step3_watermark.png` | Watermark biner 64×64 |
| `output_step4_embed.png` | Perbandingan asli vs watermarked |
| `output_step5_dct.png` | Visualisasi DCT & quantization |
| `output_step6_kompresi.png` | Foto hasil kompresi per QF |
| `output_step6_ukuran_file.png` | Bar chart ukuran file per QF |
| `output_step7_ekstraksi.png` | Watermark asli vs hasil ekstraksi |
| `output_step8_grafik_ber.png` | Grafik BER vs QF |
| `wajah_watermarked.png` | Foto hasil watermark (PNG lossless) |
| `wajah_qf10.jpg` ... `wajah_qf100.jpg` | Foto terkompresi per QF |

---

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'scipy'`**
```bash
pip install scipy
```

**Error: `File 'foto_wajah.jpg' tidak ditemukan`**
Pastikan nama file foto sama persis dengan yang ada di `FOTO_PATH`. Cek juga ekstensinya (`.jpg`, `.jpeg`, `.png`).

**Grafik tidak muncul (di server/headless)**
Tambahkan di baris paling atas script:
```python
import matplotlib
matplotlib.use('Agg')  # mode tanpa tampilan
```

**BER semua sekitar 0.49–0.50**
Kamu mungkin masih pakai gambar simulasi random, bukan foto wajah asli. Pastikan `FOTO_PATH` mengarah ke foto yang benar.
