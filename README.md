# Panduan Monitoring AMCAS via SSH
## Bus Kampus UGM — Orange Pi Zero 2W

---

## 1. Koneksi SSH ke Orange Pi

```bash
ssh orangepi@<IP_ORANGE_PI>
# contoh: ssh orangepi@192.168.1.42
```

Cek IP perangkat dari router, atau jalankan di perangkat langsung:
```bash
ip addr show | grep "inet "
```

---

## 2. Status Service ADAS

```bash
# Status singkat
systemctl status adas.service

# Mulai service
sudo systemctl start adas.service

# Hentikan service
sudo systemctl stop adas.service

# Restart service (misalnya setelah update kode)
sudo systemctl restart adas.service

# Enable/disable autostart saat boot
sudo systemctl enable  adas.service
sudo systemctl disable adas.service
```

---

## 3. Log ADAS Real-Time

```bash
# Ikuti log secara real-time (Ctrl+C untuk berhenti)
journalctl -u adas.service -f

# Log 10 menit terakhir
journalctl -u adas.service --since "10 minutes ago"

# Log sejak boot terakhir
journalctl -u adas.service -b

# Hanya ERROR dan CRITICAL
journalctl -u adas.service -p err -f

# Log dari file (hanya error yang disimpan ke file)
tail -f /var/log/adas/adas_error.log
```

**Contoh output normal:**
```
10:23:01 [INFO] main: ==================================================
10:23:01 [INFO] main: AMCAS Bus Kampus UGM — Memulai...
10:23:02 [INFO] pipeline: Model dan tracker siap.
10:23:02 [INFO] pipeline: Kamera dibuka: 256x256
10:23:02 [INFO] gps_reader: Thread dimulai (/dev/ttyS0 @ 9600 baud)
10:23:02 [INFO] lcd: Thread dimulai.
10:23:02 [INFO] buzzer: Thread dimulai (line 257, gpio_ok=True)
10:23:05 [INFO] lcd: State: SIAGA | TTC: 3.8 s | Jarak: 12.4 m
10:23:07 [INFO] lcd: State: BAHAYA | TTC: 1.6 s | Jarak: 8.2 m
```

---

## 4. System Health

### CPU, Memory, dan Load
```bash
# Monitor interaktif (Ctrl+C untuk keluar)
htop

# Satu baris ringkasan
top -b -n 1 | head -20

# Memory
free -h

# CPU info
lscpu | grep -E "CPU|Thread|Core"
```

### Suhu CPU (sangat penting di lingkungan outdoor/dalam bus)
```bash
# Suhu thermal zone 0 (core CPU) — dalam milli-Celsius
watch -n 1 cat /sys/class/thermal/thermal_zone0/temp

# Tampilkan dalam Celsius
watch -n 1 'echo "scale=1; $(cat /sys/class/thermal/thermal_zone0/temp)/1000" | bc'

# Semua thermal zone
for f in /sys/class/thermal/thermal_zone*/temp; do
  zone=$(echo $f | grep -o 'zone[0-9]')
  temp=$(cat $f)
  echo "$zone: $(echo "scale=1; $temp/1000" | bc) °C"
done
```

> **Peringatan:** H618 mulai throttle di ~85°C. Jika sering throttle,
> pasang heatsink atau aktifkan kipas cooling.

### Disk usage (flash storage terbatas)
```bash
df -h

# Ukuran log AMCAS
du -sh /var/log/adas/
```

---

## 5. Monitor Device Hardware

### Kamera USB
```bash
# Cek apakah kamera USB terdeteksi
ls /dev/v4l/by-id/
# Harus menampilkan '/dev/v4l/by-id/usb-HD_Camera_Manufacturer_HD_USB_Camera-video-index0' atau semacamnya

# Info detail kamera
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --all
```

### GPS (UART0)
```bash
# Cek apakah port tersedia
ls -la /dev/ttyS0

# Baca raw NMEA langsung (Ctrl+C untuk berhenti)
cat /dev/ttyS0

# Atau dengan stty untuk raw mode
stty -F /dev/ttyS0 9600 raw -echo && cat /dev/ttyS0
```

### GPIO dan SPI
```bash
# Cek gpiochip0
gpioinfo /dev/gpiochip0 | grep -E "line 257|line 262|line 228"

# Cek SPI device
ls -la /dev/spidev*
```

---

## 6. Deploy / Update Kode

```bash
# Salin kode baru ke Orange Pi (dari laptop developer)
rsync -avz --exclude='.git' --exclude='__pycache__' \
    ./adas/ orangepi@<IP>:/opt/adas/adas/

# Salin service file (jika berubah)
scp adas.service orangepi@<IP>:/tmp/
ssh orangepi@<IP> "sudo cp /tmp/adas.service /etc/systemd/system/ && \
                    sudo systemctl daemon-reload && \
                    sudo systemctl restart adas.service"
```

### Setup awal (pertama kali)
```bash
# Di Orange Pi Zero 2W:
sudo mkdir -p /opt/adas/models /var/log/adas
sudo python3 -m venv /opt/adas/.venv
/opt/adas/.venv/bin/pip install \
    opencv-python-headless \
    tflite-runtime \
    pyserial \
    spidev \
    gpiod

# Salin model
sudo cp best_int8.tflite /opt/adas/models/

# Install service
sudo cp adas.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable adas.service
sudo systemctl start adas.service
```

---

## 7. Troubleshooting Umum

| Gejala | Periksa |
|--------|---------|
| Service langsung restart | `journalctl -u adas.service -b -p err` |
| Kamera tidak terdeteksi | `ls /dev/v4l/by-id/` — cek apakah kamera USB muncul |
| GPS tidak dapat kecepatan | `cat /dev/ttyS0` — cek apakah ada output NMEA |
| LCD tidak menampilkan | Cek `/dev/spidev1.0` ada; cek wiring RST/DC |
| Buzzer tidak berbunyi | Cek `gpioinfo /dev/gpiochip0 \| grep 257` |
| CPU 100% / suhu tinggi | Naikkan `DETECTION_INTERVAL` di constants.py |
| Log memenuhi flash | `journalctl --vacuum-size=50M` |

---

## 8. Unggah Log untuk Debugging

```bash
# Kumpulkan log + info sistem dalam satu file
journalctl -u adas.service --since "1 hour ago" > /tmp/adas_log.txt
dmesg >> /tmp/adas_log.txt
free -h >> /tmp/adas_log.txt
df -h >> /tmp/adas_log.txt

# Unduh ke laptop
scp orangepi@<IP>:/tmp/adas_log.txt ./adas_log_$(date +%Y%m%d).txt
```

## 9. Daftar Interface Hardware

| Perangkat | Interface | Sinyal | Pin Fisik | Nama/Pemetaan |
|-----------|-----------|--------|------------|---------------|
| GPS | UART0 (`uart0`) | RX | 8 | UART0_TX / PH0 |
| GPS | UART0 (`uart0`) | TX | 10 | UART0_RX / PH1 |
| LCD TFT | SPI1 (`spidev1.0`) | MOSI | 19 | SPI1_MOSI / PH7 |
| LCD TFT | SPI1 (`spidev1.0`) | MISO | 21 | SPI1_MISO / PH8 |
| LCD TFT | SPI1 (`spidev1.0`) | CLK | 23 | SPI1_CLK / PH6 |
| LCD TFT | SPI1 (`spidev1.0`) | CS | 24 | SPI1_CS0 / PH5 |
| LCD TFT | SPI1 (`spidev1.0`) | RST | 26 | GPIO262 / PI6 |
| LCD TFT | SPI1 (`spidev1.0`) | DC | 22 | GPIO228 / PH4 |
| LCD TFT | SPI1 (`spidev1.0`) | LED | 18 | 3.3V |
| Buzzer | GPIO | Base | 12 | GPIO257 / PI1 |
| Kamera HD USB | USB (v4l2) | — | — | — |

## 10. Evaluasi Model dari Rekaman

Perintah berikut dijalankan dari `/opt/adas` pada Orange Pi.

### Ambil sampel satu frame per detik

```bash
sudo /opt/adas/.venv/bin/python3 -m adas.sample_video_frames
```

Video `*_raw.avi` di `/opt/adas/recordings` akan disampel ke:
`/opt/adas/recordings/frame_samples/<nama-video-tanpa-_raw.avi>/`.
Jika folder sampel sudah berisi file, pindahkan folder tersebut terlebih dahulu
agar sampel lama atau label tidak tercampur.

### Siapkan label dan evaluasi

Evaluasi harus memakai ground truth, bukan confidence model. Untuk setiap gambar,
siapkan file `.txt` format YOLO dengan satu baris per objek:
`class_id x_center y_center width height` (koordinat ternormalisasi 0–1).
Petakan class ID label ke kelas model: `bus,mobil,motor,truk`; sesuaikan urutan
`--class-names` dengan ID pada `data.yaml` Roboflow. Contohnya, jika ID 1 pada
Roboflow berarti `car`, isikan kelas model yang bersesuaian (`mobil`) pada
posisi kedua. Label kosong berarti gambar tersebut sudah diperiksa dan tidak
memiliki objek.

Label dapat diletakkan di samping gambar sampel atau pada folder terpisah:

```bash
sudo /opt/adas/.venv/bin/python3 -m adas.evaluate_model \
  --samples /opt/adas/recordings/frame_samples/20261029_115929 \
  --labels-dir /opt/adas/roboflow/labels/test
```

Laporan `metrics_<nama-folder-sampel>.jpg`, `.json`, `.csv`, dan `.txt`
disimpan di `/opt/adas/models`, sehingga hasil tiap video tidak saling
menimpa. Metrik utama adalah precision, recall, dan F1 pada IoU 0.5 dengan
confidence default sistem. Target precision lebih dari 80% ditandai di laporan;
confidence rata-rata saja bukan ukuran akurasi.

### Replay video dan endurance

Tanpa durasi, video diputar satu kali. Dengan durasi, video atau playlist
diulang sampai durasi tercapai. Saat endurance, perekaman output nonaktif secara
default untuk menghindari pemenuhan disk.

```bash
# Ulang satu video selama 4 jam
sudo /opt/adas/.venv/bin/python3 -m adas.main_video \
  --video /opt/adas/recordings/20261029_115929_raw.avi --duration-hours 4

# Putar semua *_raw.avi berurutan dan ulang playlist selama 4 jam
sudo /opt/adas/.venv/bin/python3 -m adas.main_video \
  --all --duration-hours 4

# Satu kali playback dengan bbox, kelas, dan confidence
sudo /opt/adas/.venv/bin/python3 -m adas.main_video \
  --video /opt/adas/recordings/20261029_115929_raw.avi --annotate
```

Output video annotated berada di `/opt/adas/recordings/pipeline_video`.
Mode video tidak menggunakan GPS, LCD, atau buzzer.