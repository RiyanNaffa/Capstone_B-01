# Panduan Kalibrasi Kamera AMCAS

Kalibrasi dilakukan **satu kali** saat development/pemasangan.  
Pengguna/driver tidak perlu melakukan ini.

---

## Yang dibutuhkan

- **Checkerboard** — cetak pola 10×7 kotak (sudut dalam 9×6) di kertas A3
  → Download: [opencv checkerboard generator](https://calib.io/pages/camera-calibration-pattern-generator)
  → Ukuran kotak: **3.2 cm** (sesuai `SQUARE_SIZE_M = 0.032` di script)
  → Tempel di papan keras/kardus agar **flat, tidak melengkung**

- **Meteran** — untuk mengukur jarak ground truth

- Kamera **sudah terpasang di posisi final** di bus (jangan dipindah setelah ini)

- Laptop/PC dengan OpenCV terinstall (bisa juga di Orange Pi langsung)

---

## Instalasi

```bash
cd calibration/
pip install -r requirements.txt
```

---

## Langkah 1 — Kalibrasi Intrinsik (focal length)

**Tujuan:** Mendapatkan `FOCAL_LENGTH_PX` yang akurat.

### Siapkan kamera
Kamera harus **diam di posisi final** di bus. Yang bergerak adalah checkerboard.

### Ambil foto

```bash
python3 capture.py
```

Tampilan live feed dari kamera akan muncul. Pegang checkerboard dan pindahkan ke berbagai posisi:

```
Posisi yang disarankan (20-30 foto):
  ✓ Tengah frame, jarak ~1m
  ✓ Tengah frame, jarak ~2m
  ✓ Sudut kiri atas
  ✓ Sudut kanan atas
  ✓ Sudut kiri bawah
  ✓ Sudut kanan bawah
  ✓ Miring ~30° ke kiri
  ✓ Miring ~30° ke kanan
  ✓ Miring ~30° ke atas
  ✓ Miring ~30° ke bawah
  ✓ Diagonal kiri-atas ke kanan-bawah
  ✓ Diagonal kanan-atas ke kiri-bawah
  ✓ Variasikan jarak 0.5m s/d 3m
```

Tekan **SPACE** untuk simpan, **Q** untuk selesai.

### Hitung kalibrasi

```bash
python3 intrinsic.py
```

Output:
```
fx = 143.25 px   ← ini yang dipakai sebagai FOCAL_LENGTH_PX
fy = 143.18 px
RMS error: 0.42 px  ← < 1.0 = bagus
```

> **RMS > 1.0?** Tambah foto, atau periksa apakah checkerboard flat dan tidak blur.

---

## Langkah 2 — Kalibrasi Ekstrinsik (pitch kamera)

**Tujuan:** Mendapatkan `CAMERA_PITCH_DEG` yang akurat.

### Siapkan setup

```
[Kamera di bus]
      ↓
      ↓  (arahkan ke depan/bawah seperti biasa)
      ↓
[Checkerboard flat di TANAH]
      ←─── ukur dengan meteran ───→  misal 5 meter
```

Checkerboard harus **flat di tanah**, tidak berdiri.

### Ukur tinggi kamera

Ukur dengan meteran dari **lantai bus** ke **lensa kamera**. Catat nilainya.

### Jalankan kalibrasi

```bash
python3 extrinsic.py
```

Script akan bertanya:
1. Tinggi kamera (meter) — masukkan hasil ukuran
2. Jarak checkerboard (meter) — misal `5.0`
3. Arahkan kamera ke checkerboard, tekan SPACE saat terdeteksi
4. Opsional: tambah pengukuran di jarak lain (misal 3m, 7m, 10m) untuk akurasi lebih tinggi

Output:
```
CAMERA_HEIGHT_M   = 1.200
CAMERA_PITCH_DEG  = 11.43
Validasi: Ground truth 5.0m → estimasi 5.03m (error 0.6%) ✓
```

---

## Langkah 3 — Update constants.py

Salin nilai hasil kalibrasi ke [`adas/constants.py`](../adas/constants.py):

```python
FOCAL_LENGTH_PX  = 143.25   # dari intrinsic.py
CAMERA_HEIGHT_M  = 1.200    # ukur fisik
CAMERA_PITCH_DEG = 11.43    # dari extrinsic.py
```

---

## Tips

| Masalah | Solusi |
|---|---|
| Checkerboard tidak terdeteksi | Pastikan cukup cahaya, tidak blur, seluruh pola terlihat |
| RMS > 1.0 | Tambah foto, periksa checkerboard tidak melengkung |
| Error jarak > 15% | Ukur ulang `CAMERA_HEIGHT_M` dengan meteran, tambah ground truth |
| Hasil pitch tidak masuk akal (< 3° atau > 25°) | Periksa orientasi checkerboard, harus flat di tanah |

## File output

| File | Isi |
|---|---|
| `calib_images/` | Foto-foto kalibrasi |
| `calib_intrinsic.npz` | K matrix, distortion, RMS |
| `calib_extrinsic.npz` | pitch, height, ground truth points |
