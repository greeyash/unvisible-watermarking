import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os

FOTO_PATH = "input/foto bila.jpeg"   
WM_SIZE   = 64                 
QF_LIST   = [10, 20, 30, 50, 70, 90, 100]  

# STEP 1 - IMPORT & CEK FILE

print("=" * 55)
print("  WATERMARKING - Sinyal & Multimedia")
print("=" * 55)
print("\n[STEP 1] Mengecek file foto...")

if not os.path.exists(FOTO_PATH):
    print(f"  ERROR: File '{FOTO_PATH}' tidak ditemukan!")
    print(f"  Pastikan foto ada di folder yang sama dengan script ini.")
    print(f"  Folder saat ini: {os.getcwd()}")
    exit()

print(f"  OK  File ditemukan: {FOTO_PATH}")

# STEP 2 - LOAD GAMBAR

print("\n[STEP 2] Load gambar sebagai grayscale...")

img = Image.open(FOTO_PATH).convert('L')
img_array = np.array(img, dtype=np.uint8)

print(f"  OK  Ukuran gambar : {img_array.shape}")
print(f"      Min pixel     : {img_array.min()}")
print(f"      Max pixel     : {img_array.max()}")
print(f"      Tipe data     : {img_array.dtype}")

plt.figure(figsize=(5, 5))
plt.imshow(img_array, cmap='gray')
plt.title('Gambar Asli (Grayscale)')
plt.axis('off')
plt.tight_layout()
plt.savefig('output_step2_gambar_asli.png', dpi=150)
plt.show()
print("  Gambar disimpan: output_step2_gambar_asli.png")

# STEP 3 - BUAT WATERMARK BINER ACAK

print("\n[STEP 3] Membuat watermark biner acak (64x64)...")

np.random.seed(42)
watermark = np.random.randint(0, 2, size=(WM_SIZE, WM_SIZE))

print(f"  OK  Shape watermark  : {watermark.shape}")
print(f"      Jumlah bit '1'   : {watermark.sum()} dari {watermark.size}")
print(f"      Contoh baris [0] : {watermark[0, :8]}")

plt.figure(figsize=(4, 4))
plt.imshow(watermark, cmap='gray')
plt.title('Watermark Biner (64x64)')
plt.axis('off')
plt.tight_layout()
plt.savefig('output_step3_watermark.png', dpi=150)
plt.show()
print("  Gambar disimpan: output_step3_watermark.png")

# STEP 4 - EMBED WATERMARK KE FOTO (METODE LSB)

print("\n[STEP 4] Menyisipkan watermark ke foto (metode LSB)...")

img_wm = img_array.copy()
img_wm[:WM_SIZE, :WM_SIZE] = (img_wm[:WM_SIZE, :WM_SIZE] & 0b11111110) | watermark
img_wm_pil = Image.fromarray(img_wm)
img_wm_pil.save('wajah_watermarked.png')

diff = np.abs(img_array.astype(int) - img_wm.astype(int))
print(f"  OK  Watermark berhasil disisipkan")
print(f"      Perbedaan max antar pixel: {diff.max()} (tidak terlihat mata)")
print(f"      File disimpan: wajah_watermarked.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Gambar Asli')
axes[0].axis('off')
axes[1].imshow(img_wm, cmap='gray')
axes[1].set_title('Gambar + Watermark (LSB)')
axes[1].axis('off')
axes[2].imshow(diff * 255, cmap='hot')
axes[2].set_title('Perbedaan (diperbesar 255x)')
axes[2].axis('off')
plt.suptitle('Penyisipan Watermark - Metode LSB', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_step4_embed.png', dpi=150)
plt.show()
print("  Gambar disimpan: output_step4_embed.png")

# STEP 5 - VISUALISASI DCT & QUANTIZATION

print("\n[STEP 5] Visualisasi DCT & Quantization JPEG...")

block = img_array[:8, :8].astype(float) - 128

def dct2(b):
    return dct(dct(b.T, norm='ortho').T, norm='ortho')

def idct2(b):
    return idct(idct(b.T, norm='ortho').T, norm='ortho')

def make_quantization_table(qf):
    base_table = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ], dtype=float)
    scale = 5000 / qf if qf < 50 else 200 - 2 * qf
    table = np.floor((base_table * scale + 50) / 100)
    return np.clip(table, 1, 255)

dct_block = dct2(block)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Visualisasi Proses DCT & Quantization JPEG per Blok 8x8', fontsize=13, fontweight='bold')

axes[0, 0].imshow(block + 128, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Blok 8x8 Asli')
axes[0, 0].axis('off')

im = axes[0, 1].imshow(np.log(np.abs(dct_block) + 1), cmap='hot')
axes[0, 1].set_title('Koefisien DCT (log scale)')
axes[0, 1].axis('off')
plt.colorbar(im, ax=axes[0, 1])

axes[0, 2].imshow(make_quantization_table(10), cmap='YlOrRd')
axes[0, 2].set_title('Quantization Table QF=10\n(nilai besar = buang banyak)')
axes[0, 2].axis('off')

axes[0, 3].imshow(make_quantization_table(100), cmap='YlOrRd')
axes[0, 3].set_title('Quantization Table QF=100\n(nilai kecil = buang sedikit)')
axes[0, 3].axis('off')

axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5, 'Setelah\nQuantization\n(nol = info hilang)',
                ha='center', va='center', fontsize=11, color='navy',
                transform=axes[1, 0].transAxes)

for col, qf in enumerate([10, 50, 100]):
    qtable = make_quantization_table(qf)
    quantized = np.round(dct_block / qtable)
    axes[1, col+1].imshow(np.abs(quantized), cmap='hot')
    axes[1, col+1].set_title(f'DCT setelah Quantization QF={qf}')
    axes[1, col+1].axis('off')

axes[2, 0].imshow(block + 128, cmap='gray', vmin=0, vmax=255)
axes[2, 0].set_title('Blok Asli (referensi)')
axes[2, 0].axis('off')

for col, qf in enumerate([10, 50, 100]):
    qtable = make_quantization_table(qf)
    quantized = np.round(dct_block / qtable)
    reconstructed = np.clip(idct2(quantized * qtable) + 128, 0, 255)
    mse = np.mean((block + 128 - reconstructed) ** 2)
    axes[2, col+1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    axes[2, col+1].set_title(f'Rekonstruksi QF={qf}\nMSE={mse:.2f}')
    axes[2, col+1].axis('off')

plt.tight_layout()
plt.savefig('output_step5_dct.png', dpi=150, bbox_inches='tight')
plt.show()
print("  OK  Gambar disimpan: output_step5_dct.png")

# ============================================================
# STEP 6 - KOMPRESI JPEG BERBAGAI QUALITY FACTOR
# ============================================================
print("\n[STEP 6] Kompresi JPEG berbagai Quality Factor...")

file_sizes = []
original_size = os.path.getsize('wajah_watermarked.png')

print(f"  {'QF':<6} {'Ukuran File':<15} {'Rasio Kompresi'}")
print("  " + "-" * 38)

for qf in QF_LIST:
    path = f'wajah_qf{qf}.jpg'
    img_wm_pil.save(path, 'JPEG', quality=qf)
    size = os.path.getsize(path)
    file_sizes.append(size)
    ratio = original_size / size
    print(f"  QF={qf:<4} {size/1024:.1f} KB          {ratio:.1f}x lebih kecil")

fig, axes = plt.subplots(1, len(QF_LIST), figsize=(24, 4))
fig.suptitle('Hasil Kompresi JPEG per Quality Factor', fontsize=13, fontweight='bold')
for i, qf in enumerate(QF_LIST):
    img_qf = np.array(Image.open(f'wajah_qf{qf}.jpg').convert('L'))
    axes[i].imshow(img_qf, cmap='gray')
    axes[i].set_title(f'QF={qf}\n{file_sizes[i]/1024:.1f} KB')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('output_step6_kompresi.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(QF_LIST, [s/1024 for s in file_sizes], color='steelblue', alpha=0.7, width=6)
plt.title('Ukuran File vs Quality Factor')
plt.xlabel('Quality Factor (QF)')
plt.ylabel('Ukuran File (KB)')
plt.grid(True, alpha=0.3, axis='y')
for qf, size in zip(QF_LIST, file_sizes):
    plt.text(qf, size/1024 + 0.5, f'{size/1024:.1f}KB', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('output_step6_ukuran_file.png', dpi=150)
plt.show()
print("  OK  Gambar disimpan: output_step6_kompresi.png & output_step6_ukuran_file.png")

# ============================================================
# STEP 7 - EKSTRAKSI WATERMARK & PERHITUNGAN BER
# ============================================================
print("\n[STEP 7] Ekstraksi watermark & perhitungan BER...")

def extract_watermark(image_path, wm_shape=(64, 64)):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    return arr[:wm_shape[0], :wm_shape[1]] & 1

def hitung_ber(original_wm, extracted_wm):
    errors = np.sum(original_wm != extracted_wm)
    return errors / original_wm.size

ber_values = []
print(f"  {'QF':<6} {'BER':<10} {'Keterangan'}")
print("  " + "-" * 38)

for qf in QF_LIST:
    extracted = extract_watermark(f'wajah_qf{qf}.jpg')
    error_rate = hitung_ber(watermark, extracted)
    ber_values.append(error_rate)
    status = 'Bisa diekstrak' if error_rate < 0.1 else 'Tidak bisa diekstrak'
    print(f"  QF={qf:<4} BER={error_rate:.4f}   {status}")

fig, axes = plt.subplots(2, len(QF_LIST), figsize=(24, 7))
fig.suptitle('Watermark Asli vs Hasil Ekstraksi per Quality Factor', fontsize=13, fontweight='bold')
for i, qf in enumerate(QF_LIST):
    axes[0, i].imshow(watermark, cmap='gray')
    axes[0, i].set_title(f'Watermark Asli\n(QF={qf})')
    axes[0, i].axis('off')
for i, qf in enumerate(QF_LIST):
    extracted = extract_watermark(f'wajah_qf{qf}.jpg')
    axes[1, i].imshow(extracted, cmap='gray')
    axes[1, i].set_title(f'Ekstraksi QF={qf}\nBER={hitung_ber(watermark, extracted):.4f}')
    axes[1, i].axis('off')
plt.tight_layout()
plt.savefig('output_step7_ekstraksi.png', dpi=150)
plt.show()
print("  OK  Gambar disimpan: output_step7_ekstraksi.png")


# STEP 8 - GRAFIK BER VS QUALITY FACTOR

print("\n[STEP 8] Membuat grafik BER vs Quality Factor...")

plt.figure(figsize=(9, 5))
plt.plot(QF_LIST, ber_values, marker='o', color='royalblue', linewidth=2, label='BER')
plt.axhline(y=0.1, color='red', linestyle='--', label='Threshold BER=0.1')
plt.fill_between(QF_LIST, ber_values, 0.1,
                 where=[b > 0.1 for b in ber_values],
                 alpha=0.1, color='red', label='Zona tidak bisa diekstrak')
plt.title('BER vs Quality Factor JPEG', fontsize=13, fontweight='bold')
plt.xlabel('Quality Factor (QF)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output_step8_grafik_ber.png', dpi=150)
plt.show()

print("\n" + "=" * 55)
print("  KESIMPULAN")
print("=" * 55)
for qf, bv in zip(QF_LIST, ber_values):
    status = 'Bisa diekstrak    ✓' if bv < 0.1 else 'Tidak bisa diekstrak ✗'
    print(f"  QF={qf:<4} -> BER={bv:.4f} -> {status}")

gagal = [qf for qf, bv in zip(QF_LIST, ber_values) if bv >= 0.1]
if gagal:
    print(f"\n  Watermark tidak bisa diekstrak pada QF: {gagal}")
else:
    print(f"\n  Watermark bisa diekstrak di semua QF yang diuji.")

print("\n  File output tersimpan di folder yang sama.")
print("=" * 55)
