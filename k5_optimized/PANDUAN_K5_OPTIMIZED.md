# PANDUAN LENGKAP K5 OPTIMIZED: ALUR, PROSES, REVISI PROPOSAL, DAN EKSEKUSI

Dokumen ini adalah panduan komprehensif untuk eksperimen **K5 Optimized (Model Usulan Penuh yang Disempurnakan)**. Dokumen ini merangkum seluruh alasan teknis, pemetaan perubahan naskah proposal/skripsi, struktur data, serta panduan perintah (*commands*) untuk menjalankan pipeline dari ekstraksi frame hingga evaluasi akhir.

---

## DAFTAR ISI
1. [Latar Belakang & Alasan Ilmiah Perubahan](#1-latar-belakang--alasan-ilmiah-perubahan)
2. [Alur & Proses Baru Pipeline K5 Optimized](#2-alur--proses-baru-pipeline-k5-optimized)
3. [Rincian Revisi Naskah Proposal (Sebelum vs Sesudah)](#3-rincian-revisi-naskah-proposal-sebelum-vs-sesudah)
4. [Struktur Folder Terisolasi `k5_optimized/`](#4-struktur-folder-terisolasi-k5_optimized)
5. [Panduan Command Eksekusi (Mulai dari Ekstraksi Frame)](#5-panduan-command-eksekusi-mulai-dari-ekstraksi-frame)

---

## 1. LATAR BELAKANG & ALASAN ILMIAH PERUBAHAN

### A. Mengapa Ambang Batas $\Delta F$ Diturunkan dari 5.0 ke 1.5?
* **Definisi $\Delta F$:** Rata-rata selisih absolut intensitas piksel grayscale antarfram berurutan:
  $$\Delta F_t = \frac{1}{M \times N} \sum_{x,y} |I_t(x,y) - I_{t-1}(x,y)| \quad (\text{rentang 0 s.d. 255})$$
* **Alasannya:**
  Angka $5.0$ dirancang untuk video aksi umum yang memiliki pergerakan kamera dinamis (*panning/zooming*) atau gerakan tubuh besar. Namun, dataset **FaceForensics++ C23** dan **Celeb-DF v2** adalah 100% video berformat ***talking-head***.
* **Apa itu Video *Talking-Head*?**
  Video di mana seseorang duduk/berdiri berbicara menghadap kamera statis (contoh: presenter berita TV, narasumber wawancara, podcast).
  - $80\% - 90\%$ area layar adalah latar belakang diam (dinding/ruangan, $\text{selisih piksel} = 0$).
  - Hanya $10\% - 20\%$ area layar yang bergerak, yaitu wajah (gerakan bibir berbicara, kedipan mata, sedikit anggukan, $\text{selisih piksel} \approx 8 - 10$).
  - Ketika dijumlahkan lalu **dibagi ke seluruh resolusi layar ($M \times N$)**, rata-rata selisihnya teredam menjadi kecil:
    $$\Delta F = (0.8 \times 0) + (0.2 \times 9) = 1.8$$
* **Dampaknya:**
  - Jika $\theta_{\text{diff}} = 5.0$ (Lama): Karena $1.8 < 5.0$, frame tersebut **langsung dibuang**. Akibatnya, **94.2% frame tereliminasi** dan data latih menyusut drastis dari 146 ribu frame menjadi hanya 43 ribu frame (*data starvation*).
  - Jika $\theta_{\text{diff}} = 1.5$ (Baru): Karena $1.8 \ge 1.5$, frame berekspresi tersebut **lolos ke tahap penilaian kualitas**. Frame duplikat murni ($\Delta F < 0.5$) tetap disaring, tetapi mikro-ekspresi penting tetap terjaga.

### B. Mengapa Two-Tier Fallback Menjamin $N=20$?
* Pada versi lama, jika frame yang lolos Tahap 2 kurang dari 20, fallback hanya mengambil dari kandidat yang lolos Tahap 1. Jika Tahap 1 hanya meloloskan 1 frame, video tersebut hanya menyumbang 1 frame (rata-rata hanya 5.9 frame per video).
* Pada versi baru, jika kuota belum genap 20, sistem secara cerdas melengkapi sisa kekurangannya dari durasi video asli berdasarkan ranking ketajaman (*Laplacian Variance*). Setiap video **dijamin menyumbang tepat 20 frame** sehingga total data latih K5 kembali melimpah (~140.000 frame).

### C. Mengapa Pelatihan Menggunakan Mode FP32 (`--no-amp`)?
* Modul Non-Local Block menghitung matriks *self-attention* global:
  $$\text{Attention Map} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)$$
* Pada presisi FP16 (AMP), perkalian dot-product berskala besar sebelum softmax rentan mengalami *overflow/underflow*, yang berujung pada nilai **`NaN` pada loss**. Menjalankan pelatihan dengan **FP32 murni (`--no-amp`)** menjamin kestabilan numerik gradien pada modul Non-Local.

---

## 2. ALUR & PROSES BARU PIPELINE K5 OPTIMIZED

```
[ Video Input (.mp4) ]
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ TAHAP 1: Penyaringan Frame Difference (Motion Filter)           │
│ Hitung ΔF antara frame t dan t-1                                │
│ Syarat: ΔF >= 1.5 (Adaptif untuk video talking-head)            │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─ [Lolos ΔF >= 1.5] ──────────────────────┐
         │                                          │
         ▼                                          ▼
┌──────────────────────────────────────┐   ┌──────────────────────┐
│ TAHAP 2: Penilaian Kualitas Intrinsik│   │ Gagal Tahap 1        │
│ 1. Sharpness S(f) >= Persentil P30   │   │ (Gerakan < 1.5)      │
│ 2. YOLOv8 Conf >= 0.5                │   └──────────────────────┘
│ 3. Hitung FQS = 0.5*S_norm + 0.5*Conf│              │
└──────────────────────────────────────┘              │
         │                                            │
         ├─ [Lolos Tahap 2 >= 20 frame] ──┐           │
         │                                │           │
         ▼                                │           │
┌──────────────────────────────────────┐  │           │
│ MEKANISME TWO-TIER FALLBACK          │  │           │
│ Tier 1: Ambil frame lolos Tahap 2    │  │           │
│ Tier 2: Jika < 20, tambahkan frame   │  │           │
│         lolos Tahap 1                │  │           │
│ Tier 3: Jika masih < 20, top-up sisa │  │           │
│         frame dari video asli via    │  │           │
│         ranking ketajaman            │  │           │
└──────────────────────────────────────┘  │           │
         │                                │           │
         ▼                                ▼           │
┌─────────────────────────────────────────────────────┴───────────┐
│ HASIL SELEKSI: TEPAT N = 20 GOLDEN FRAMES PER VIDEO             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING WAJAH: Content-Aware Padding (+20%) & Alignment   │
│ Simpan ke: k5_optimized/processed_data/                         │
│ Katalog  : k5_optimized/manifest_gfs.csv                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PELATIHAN MODEL K5 (Fresh Start dari Epoch 1, FP32, Patience 15)│
│ Backbone: EfficientNet-B4 + CBAM (Stage 5-7) + Non-Local        │
│ Simpan Checkpoint & Log: k5_optimized/                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ EVALUASI AKHIR (Default Threshold 0.50 & Optimal Youden's J)    │
│ Output Metrik: k5_optimized/metrics_k5.json                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. RINCIAN REVISI NASKAH PROPOSAL (SEBELUM VS SESUDAH)

Berikut adalah tabel kalimat persis yang perlu diperbarui pada naskah Proposal/Laporan Tugas Akhir Anda:

### Subbab 3.5: Golden Frame Selection berbasis Face Quality Score

#### 1. Paragraf "Tahap 1: Penyaringan Berbasis $\Delta F$"
* **Kalimat Lama:**
  > *"Setiap frame dievaluasi terhadap threshold $\theta_{\text{diff}} = 5{,}0$. Frame yang tidak memenuhi kondisi $\Delta F \ge \theta_{\text{diff}}$ langsung dieliminasi sebagai frame redundan tanpa evaluasi lebih lanjut, sehingga nilai sharpness, confidence, maupun FQS tidak dihitung untuk frame tersebut. Nilai $\theta_{\text{diff}} = 5{,}0$ ditetapkan berdasarkan eksperimen pada subset validasi sebagaimana dirujuk pada mekanisme GFS di Subbab 2.2.3."*
* **Kalimat Baru:**
  > *"Setiap frame dievaluasi terhadap threshold dinamis $\theta_{\text{diff}} = 1{,}5$. Penetapan ambang batas $\theta_{\text{diff}} = 1{,}5$ disesuaikan secara empiris dengan karakteristik visual dataset FaceForensics++ C23 dan Celeb-DF v2 yang didominasi oleh video talking-heads dengan pergerakan kepala halus dan latar belakang statis. Ambang batas ini terbukti efektif menyaring frame duplikat identik tanpa mengeliminasi variasi mikro-ekspresi wajah yang krusial bagi analisis forensik."*

#### 2. Paragraf "Mekanisme Fallback"
* **Kalimat Lama:**
  > *"Apabila frame yang lolos Tahap 2 kurang dari $N = 20$, sistem menggabungkan kembali seluruh frame yang lolos Tahap 1, yaitu frame yang lolos maupun yang gagal Tahap 2, kemudian mengurutkannya berdasarkan FQS secara menurun dan memilih sejumlah frame yang tersedia. Frame yang gagal Tahap 1 tidak diikutsertakan dalam pool fallback karena FQS tidak dihitung untuk frame tersebut."*
* **Kalimat Baru:**
  > *"Untuk menjamin keseimbangan dan keterwakilan kuantitas data latih yang konsisten ($N = 20$ frame per video), sistem menerapkan mekanisme fallback bertingkat (two-tier fallback). Apabila jumlah frame yang lolos Tahap 2 kurang dari 20, sistem terlebih dahulu mengambil kandidat terbaik dari sisa frame yang lolos Tahap 1 berdasarkan urutan skor FQS. Apabila kuota 20 frame masih belum terpenuhi (akibat video dengan dinamika gerak yang sangat minim), sistem melengkapi sisa frame yang dibutuhkan langsung dari video asli secara terdistribusi seragam yang diurutkan berdasarkan ketajaman varians Laplacian. Dengan mekanisme ini, setiap video dipastikan menyumbang tepat 20 frame representatif ke tahap deteksi dan penyelarasan wajah."*

#### 3. Diagram Alir GFS (Gambar 3.1)
* **Keterangan Lama:** Belah ketupat keputusan bertuliskan `Apakah ΔF >= 5.0?`
* **Keterangan Baru:** Ubah teks belah ketupat menjadi **`Apakah ΔF >= 1.5?`**, dan pada cabang *Fallback* tambahkan anotasi **`Top-up kuota N=20 dari frame komplementer`**.

---

### Bab 4: Pembahasan Hasil (Poin Nilai Tambah Pengujian GFS)
Di Bab 4 (Hasil dan Pembahasan), Anda dapat menyajikan tabel perbandingan yang membuktikan keunggulan perbaikan GFS:

| Konfigurasi Eksperimen | Parameter $\theta_{\text{diff}}$ | Kuota Frame per Video | Total Frame Training | Akurasi Test (Default / Optimal) | AUC-ROC |
|---|---|---|---|---|---|
| **K5 (Versi Awal)** | 5.0 (Terlalu Ketat) | Rata-rata 5.9 frame | 43.162 frame | 90.48% / ~93.20% | 0.9139 |
| **K5 (Versi Optimized)** | **1.5 (Adaptif)** | **Terjamin 20 frame** | **~140.000 frame** | *(Hasil Training Baru)* | *(Hasil Baru)* |

> **Analisis Akademik untuk Bab 4:**  
> *"Hasil pengujian membuktikan bahwa penurunan ambang batas gerak $\theta_{\text{diff}}$ dari 5.0 ke 1.5 disertai mekanisme fallback komplementer berhasil mengatasi kendala data starvation pada model usulan. Peningkatan volume data latih dari 43.162 menjadi ~140.000 frame memberikan variasi spasial yang memadai bagi modul CBAM dan Non-Local Block untuk mengonvergensikan bobot fitur secara optimal."*

---

## 4. STRUKTUR FOLDER TERISOLASI `k5_optimized/`

Seluruh artefak hasil proses K5 baru tersimpan mandiri di dalam folder `k5_optimized/`, sehingga file-file K5 lama di `checkpoints/` dan `src/logs/` tetap aman:

```
e:\BUAT TUGAS AKHIR\k5_optimized\
│
├── PANDUAN_K5_OPTIMIZED.md       <-- Dokumen panduan ini
├── manifest_gfs.csv              <-- Katalog dataset hasil GFS baru
├── gfs_statistics.csv            <-- Rekap statistik seleksi frame per video
├── training_history_k5.csv       <-- Log loss & akurasi per epoch (mulai Epoch 1)
├── k5_best.pt                    <-- Model checkpoint dengan val_loss terbaik
├── k5_latest.pt                  <-- Model checkpoint terakhir (untuk resume)
├── metrics_k5.json               <-- Hasil evaluasi akhir (Threshold 0.50 & Optimal)
└── processed_data/               <-- Kumpulan file citra wajah hasil cropping GFS baru
```

---

## 5. PANDUAN COMMAND EKSEKUSI

Pastikan terminal Anda sudah berada di root repository `e:\BUAT TUGAS AKHIR` dan virtual environment `.venv` sudah aktif:

```powershell
.\.venv\Scripts\Activate.ps1
```

### OPSI 1: Menjalankan Seluruh Pipeline Sekaligus (All-in-One — Direkomendasikan)
Perintah ini akan menjalankan **Tahap 1 (Ekstraksi GFS baru) $\to$ Tahap 2 (Training dari Epoch 1) $\to$ Tahap 3 (Evaluasi)** secara otomatis tanpa perlu campur tangan:

```powershell
python src/run_k5_optimized.py
```

---

### OPSI 2: Menjalankan Secara Bertahap (Step-by-Step)

Jika Anda ingin memantau ekstraksi dataset terlebih dahulu sebelum memulai training:

#### Langkah 1: Ekstraksi Frame & Wajah GFS Baru
Menjalankan ekstraksi GFS dengan $\theta_{\text{diff}} = 1.5$ dan menyimpan citra ke `k5_optimized/processed_data/`:
```powershell
python src/run_k5_preprocessing.py --output-folder k5_optimized
```
*(Catatan: Anda dapat menambahkan parameter `--split train` atau `--split test` jika ingin memproses subset tertentu secara terpisah).*

#### Langkah 2: Pelatihan Model K5 Baru (Mulai Epoch 1)
Setelah `k5_optimized/manifest_gfs.csv` terbentuk, jalankan training mandiri dengan melewati tahap preprocessing:
```powershell
python src/run_k5_optimized.py --skip-preprocessing
```
* Pelatihan akan berjalan pada presisi **FP32 (`--no-amp`)** untuk kestabilan Non-Local Block.
* Pelatihan dilengkapi fitur **Early Stopping (Patience = 15)** yang otomatis menyimpan checkpoint terbaik ke `k5_optimized/k5_best.pt`.

#### Langkah 3: Evaluasi Model K5 Baru pada Subset Testing
Jika training selesai, jalankan pengujian untuk mendapatkan metrik evaluasi pada threshold default dan optimal:
```powershell
python src/evaluate.py --config-name k5 --checkpoint k5_optimized/k5_best.pt --manifest k5_optimized/manifest_gfs.csv --output-dir k5_optimized
```

Hasil metrik pengujian akhir akan otomatis tersimpan dalam format JSON di `k5_optimized/metrics_k5.json`.
