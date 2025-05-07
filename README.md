```
> NAMA KELOMPOK   : Kelompok 13
> NAMA SUPERVISOR :
    - SAKTI ARISENA DAMAI PRASETYO (2042221029)
    - YASMINE YULIANA SALIM (2042221062)
    - AULIAZQI RARASATI NAJID (2042221116)

> NAMA DEPARTEMEN / INSTITUT : Teknik Instrumentasi / INSTITUT TEKNOLOGI SEPULUH NOPEMBER

```

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

Sumber Refrensi :
- https://www.kaggle.com/code/satyaprakashshukl/rice-classification
  
[1]	K. H. Ng, S. C. Liew, and F. Ernawan, “An Improved Image Steganography Scheme Based on RDWT and QR Decomposition,” IOP Conf. Ser. Mater. Sci. Eng., vol. 769, no. 1, pp. 222–231, 2020. 

[2]	B. Ando, S. Baglio, S. Castorina, R. Crispino, and V. Marletta, “A Methodology for the Development of Low-Cost, Flexible Touch Sensor for Application in Assistive Technology,” IEEE Trans. Instrum. Meas., vol. 70, 2021. 

[3]	V. Krishnasamy and S. Venkatachalam, “An efficient data flow material model based cloud authentication data security and reduce a cloud storage cost using Index-level Boundary Pattern Convergent Encryption algorithm,” Mater. Today Proc., vol. 81, no. 2, pp. 931–936, 2021. 

[4]	X. Yang et al., “A Survey on Smart Agriculture: Development Modes, Technologies, and Security and Privacy Challenges,” IEEE/CAA J. Autom. Sin., vol. 8, no. 2, pp. 273–302, 2021. 

[5]	S. Ibrahim, S. B. A. Kamaruddin, A. Zabidi, and N. A. M. Ghani, “Contrastive analysis of rice grain classification techniques: Multi-class support vector machine vs artificial neural network,” IAES Int. J. Artif. Intell., vol. 9, no. 4, pp. 616–622, 2020. 

[6]	A. S. Hamzah and A. Mohamed, “Classification of white rice grain quality using ann: A review,” IAES Int. J. Artif. Intell., vol. 9, no. 4, pp. 600–608, 2020. 

[7]	MUH ZAINAL ALTIM, FAISAL, SALMIAH, KASMAN, ANDI YUDHISTIRA, and RITA AMALIA SYAMSU, “Pengklasifikasi Beras Menggunakan Metode Cnn (Convolutional Neural Network),” J. INSTEK (Informatika Sains dan Teknol., vol. 7, no. 1, pp. 151–155, 2022. 

[8]	P. S. Sampaio, A. S. Almeida, and C. M. Brites, “Use of artificial neural network model for rice quality prediction based on grain physical parameters,” Foods, vol. 10, no. 12, 2021. 

[9]	W. Xia, R. Peng, H. Chu, X. Zhu, Z. Yang, and ..., “An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice,” Available SSRN …. 

[10]	A. Bhattacharjee, K. R. Singh, T. S. Singh, S. Datta, U. R. Singh, and G. Thingbaijam, “INTELLIGENT SYSTEMS AND APPLICATIONS IN ENGINEERING A Comparative Study on Rice Grain Classification Using Convolutional Neural Network and Other Machine Learning Techniques,” pp. 0–1, 2024. 

[11]	T. T. H. Phan, Q. T. Vo, and H. Du Nguyen, “A novel method for identifying rice seed purity using hybrid machine learning algorithms,” Heliyon, vol. 10, no. 14, 2024. 

[12]	Y. Wang, H. Wang, and Z. Peng, “Rice diseases detection and classification using attention based neural network and bayesian optimization,” Expert Syst. Appl., vol. 178, 2021. 

[13]	Y. Haddad, K. Salonitis, and C. Emmanouilidis, “A decision-making framework for the design of local production networks under largescale disruptions,” Procedia Manuf., vol. 55, no. C, pp. 393–400, 2021. 

[14]	I. Samarakoon and P. Liyanage, “Impact of Data Analytics on Operations Success of Apparel Sector ABC Clothing Pvt Limited (Sri Lanka),” Int. J. Comput. Appl., vol. 184, no. 33, pp. 1–15, 2022. 

[15]	Q. W. Kong, J. He, Z. W. Zhang, H. Zheng, and P. Z. Wang, “Projection as a way of thinking to find factors in factor space,” Procedia Comput. Sci., vol. 199, pp. 503–508,

---
