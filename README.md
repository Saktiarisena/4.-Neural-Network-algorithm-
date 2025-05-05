# 🌾 Rice Classification with Multi-Layer Neural Network in Rust

Proyek ini merupakan implementasi sederhana dari **Multi-Layer Neural Network (MLP)** untuk melakukan klasifikasi varietas beras berdasarkan fitur morfologi seperti *solidity*, *aspect ratio*, *roundness*, dan *compactness*. Proyek ini ditulis dalam bahasa pemrograman **Rust** menggunakan `ndarray` dan `linfa`.

---

## 📂 Dataset

Dataset yang digunakan adalah file CSV bernama `Rice_MSC_Dataset_sample.csv`, dengan kolom-kolom:

* `Solidity`
* `Aspect_Ratio`
* `Roundness`
* `Compactness`
* `Class` (label kelas, seperti "Jasmine", "Karacadag", dll.)

---

## 🧠 Arsitektur Neural Network

Model neural network yang digunakan terdiri dari:

* 4 input neurons (sesuai jumlah fitur)
* 1 hidden layer dengan 64 neurons dan aktivasi ReLU
* Output layer sesuai jumlah kelas (one-hot encoding)
* Optimisasi menggunakan *gradient descent* manual

---

## 🚀 Fitur Program

* Membaca dataset dari file `.csv`
* Mengubah label kelas menjadi indeks numerik menggunakan one-hot encoding
* Melatih neural network dengan forward dan backward propagation
* Menampilkan akurasi prediksi dan hasil klasifikasi
* Progres pelatihan ditampilkan dengan animasi titik berjalan

---

## 📦 Dependencies

Tambahkan dependencies berikut di `Cargo.toml`:

```toml
[dependencies]
linfa = "0.7.1"
linfa-svm = "0.7.2"
linfa-nn = "0.7.1"
linfa-clustering = "0.7.1"
ndarray = "0.15.6"
csv = "1.1"
rand = "0.9.0"
plotters = "0.3.0"
linfa-logistic = "0.7.0"
```

---

## 🛠️ Cara Menjalankan

1. Pastikan Anda memiliki Rust terinstal. Jika belum:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Clone repositori ini:

   ```bash
   git clone https://github.com/username/rice-nn-rust.git
   cd rice-nn-rust
   ```

3. Letakkan file `Rice_MSC_Dataset_sample.csv` ke dalam folder `data/`.

4. Jalankan program:

   ```bash
   cargo run
   ```

---

## 📊 Output

* Menampilkan fitur dan peta kelas
* Menampilkan akurasi dari model
* Contoh prediksi:

```
Sample 0: Predicted 2, Actual 2
Sample 1: Predicted 0, Actual 0
...
Accuracy: 96.67%
```

---
