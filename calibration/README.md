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

- Kamera yang sama dengan kamera yang akan dipakai sistem

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
Untuk kalibrasi intrinsik, kamera boleh berada di mana saja; tidak perlu
dipasang di bus atau diarahkan pada sudut final. Gunakan kamera, lensa, fokus,
zoom, resolusi, dan pemrosesan gambar (crop/resize) yang sama dengan sistem
nantinya. Saat mengambil foto, kamera harus diam dan checkerboard yang
dipindahkan.

### Ambil foto

Stop service ADAS terlebih dahulu agar service dan script kalibrasi tidak
mengakses kamera secara bersamaan:

```bash
sudo systemctl stop adas.service
```

Jalankan server preview dari folder `calibration`:

```bash
cd calibration/
python3 capture.py
```

Di browser pada Orange Pi, buka `http://127.0.0.1:8080`. Untuk membuka preview
dari perangkat lain di jaringan lokal, jalankan server dengan:

```bash
python3 capture.py --host 0.0.0.0
```

Lalu buka `http://<IP_ORANGE_PI>:8080`. Gunakan hanya di jaringan tepercaya;
halaman capture tidak memakai autentikasi. Gunakan tombol **Capture frame** di
halaman web untuk menyimpan gambar. Tekan **Ctrl+C** di terminal untuk berhenti.
Preview berjalan sebagai MJPEG stream dan tidak memerlukan desktop/Qt.

Jika gambar terlalu terang, centang **Use manual exposure**, turunkan nilai
**Exposure**, lalu tekan **Apply camera settings**. Turunkan **Brightness**
juga jika diperlukan. Nilai dan rentang kontrol bergantung pada kamera/driver;
periksa nilai readback dan perubahan di preview. Jika driver menolak kontrol,
gunakan utilitas V4L2 (`v4l2-ctl --list-ctrls`) untuk melihat kontrol yang
didukung kamera.

Posisikan checkerboard supaya seluruh pola terlihat dan tajam. Variasikan posisi
dan kemiringannya:

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

Ekstrinsik memakai preview MJPEG berbasis web (tanpa `imshow`/Qt). Secara
default buka `http://127.0.0.1:8081` pada Pi. Untuk mengakses dari perangkat
lain di jaringan tepercaya, jalankan:

```bash
python3 extrinsic.py --host 0.0.0.0
```

Kemudian buka `http://<IP_ORANGE_PI>:8081`. Halaman tidak memakai autentikasi;
jangan jalankan pada jaringan yang tidak tepercaya. Tempatkan checkerboard flat
di tanah sesuai jarak yang dimasukkan, tunggu overlay sudut terdeteksi, lalu
tekan **Capture detected checkerboard**. Pengukuran jarak tambahan dilakukan
dengan cara yang sama; klik **Cancel current capture** bila tidak ingin
menambahkan pengukuran pada sesi tersebut. Tekan **Ctrl+C** di terminal untuk
menghentikan kalibrasi.

Script akan bertanya:
1. Tinggi kamera (meter) — masukkan hasil ukuran
2. Jarak checkerboard (meter) — misal `5.0`
3. Buka preview browser, arahkan kamera ke checkerboard, lalu capture saat terdeteksi
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

## Visualisasi IPM dari gambar

Setelah tersedia gambar dashcam, script di root repository dapat membuat
visualisasi bird's-eye view menggunakan parameter kamera di
`adas/constants.py`:

```bash
python3 ipm_image.py path/to/dashcam.jpg
```

Default area yang ditampilkan mulai dari jarak tanah yang terlihat di bagian
bawah gambar hingga 30 m ke depan, dengan lebar 12 m. Horizon secara teoritis
berjarak tak terhingga, sehingga jarak terjauh harus dibatasi. ROI dapat diubah:

```bash
python3 ipm_image.py path/to/dashcam.jpg --near-m 3 --far-m 20 --width-m 8
```

Area ground yang tidak terlihat pada gambar sumber akan tampak hitam. Hasil
adalah visualisasi berdasarkan model kamera pinhole; distorsi lensa tidak
dihilangkan.
