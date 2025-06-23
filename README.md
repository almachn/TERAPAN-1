# Analisis Faktor Penentu Popularitas Lagu dan Tren Audionya dari Masa ke Masa (Studi Kasus: Dataset Spotify 1921-2020)

## Domain Proyek 
Proyek ini mengeksplorasi dataset lagu Spotify dari tahun 1921 hingga 2020 dengan berbagai fitur akustik, tren popularitas, dan karakteristik audio

### Latar Belakang
Musik telah menjadi bagian tak terpisahkan dari kehidupan manusia—bukan hanya sebagai bentuk hiburan, tetapi juga sebagai media ekspresi, identitas budaya, bahkan alat politik dan sosial. Dalam satu abad terakhir, industri musik mengalami perubahan yang sangat masif, baik dari sisi gaya, teknologi produksi, hingga cara distribusinya. Dari piringan hitam ke kaset, dari CD ke MP3, hingga kini mendominasi melalui layanan streaming digital seperti Spotify.

Di era streaming, data adalah raja. Spotify, sebagai salah satu platform musik terbesar di dunia, menyimpan jejak digital dari jutaan lagu dan interaksinya dengan pengguna. Setiap lagu yang diunggah ke platform ini tidak hanya terdiri dari judul dan artisnya saja, tetapi juga diperkaya dengan beragam fitur audio: mulai dari danceability, energy, acousticness, valence, hingga tempo dan liveness. Fitur-fitur ini dihasilkan dari pemrosesan sinyal digital dan menjadi representasi numerik dari karakteristik suara lagu tersebut.

Namun, tidak semua lagu mendapatkan popularitas yang sama. Sebagian lagu bisa merajai tangga lagu, viral di media sosial, dan menjadi bagian dari budaya pop; sementara sebagian lainnya nyaris tidak terdengar. Lalu pertanyaannya: apa sebenarnya yang membuat sebuah lagu menjadi populer? Apakah ada pola tersembunyi dalam data audio yang bisa kita ungkap? Apakah ada karakteristik tertentu yang menjadi tren pada periode-periode waktu tertentu?

Dengan memanfaatkan dataset Spotify dari tahun 1921 hingga 2020 yang berisi lebih dari 170.000 lagu, kita memiliki peluang langka untuk menggali lebih dalam tentang evolusi musik selama hampir satu abad. Tidak hanya itu, kita juga dapat membangun model prediktif yang dapat membantu memetakan kemungkinan popularitas sebuah lagu berdasarkan fitur-fiturnya.

Studi ini menjadi semakin relevan di tengah era di mana industri musik semakin data-driven. Produser, label, bahkan musisi independen kini sangat terbantu dengan insight berbasis data dalam menentukan gaya produksi, susunan musik, hingga strategi perilisan. Oleh karena itu, penelitian ini tidak hanya memiliki nilai teoritis dalam dunia data science dan musikologi, tetapi juga nilai praktis yang signifikan dalam pengambilan keputusan industri hiburan.

Melalui pendekatan eksploratif dan pemodelan machine learning, kita dapat memberikan gambaran menyeluruh mengenai bagaimana musik populer berubah dari masa ke masa dan apa yang membentuknya. Dengan begitu, kita bisa menjawab pertanyaan penting: “Seperti apa lagu yang disukai dunia?” dan “Apakah kita bisa menciptakan hit berikutnya hanya dari angka-angka?”
## Business Understanding

### Problem Statement

Popularitas lagu adalah sesuatu yang kompleks dan dipengaruhi oleh banyak faktor. Namun, dengan adanya ratusan ribu lagu dan puluhan fitur audio, sulit untuk secara manual menentukan pola umum yang membedakan lagu populer dari yang kurang terdengar. Bagaimana kita bisa mengidentifikasi fitur-fitur kunci yang mendorong popularitas suatu lagu, dan bagaimana tren ini berubah dari waktu ke waktu?

### Goal

1. Mengidentifikasi fitur-fitur audio apa yang paling berkontribusi terhadap popularitas lagu.
2. Menganalisis bagaimana tren musik berubah dari tahun ke tahun berdasarkan fitur akustik.
3. Membangun model prediksi popularitas lagu berdasarkan fitur-fiturnya.
4. Memberikan wawasan strategis bagi industri musik berbasis data historis.

### Solution Statements

**1. Menggunakan algoritma machine learning regresi (Linear Regression, Random Forest, Gradient Boosting, XGBoost) untuk memprediksi tingkat popularitas.**
Linear Regression (Sebagai Baseline):

Regresi linier (Linear Regression) adalah salah satu algoritma dasar dalam pembelajaran mesin yang digunakan untuk memodelkan hubungan antara satu atau lebih variabel independen (karakteristik) dan variabel dependen (tujuan) yang bersifat kontinu. Tujuan algoritma ini adalah menemukan garis regresi terbaik yang dapat memprediksi nilai tujuan berdasarkan nilai input.

Dalam regresi linier sederhana (satu variabel independen), hubungan antara input (X) dan output (Y) dimodelkan dalam bentuk persamaan linier:

Y=aX+b

di mana:

    a adalah koefisien regresi yang menunjukkan kemiringan garis (slope),

    b adalah intercept (titik potong dengan sumbu Y).

Untuk kasus dengan lebih dari satu variabel input (regresi linear berganda), persamaan diperluas menjadi:

Y=a1​X1​+a2​X2​+…+an​Xn​+b

Algoritma ini menggunakan metode Ordinary Least Squares (OLS) untuk meminimalkan selisih kuadrat (mean squared error) antara nilai prediksi dan nilai aktual, sehingga diperoleh model yang paling sesuai dengan data pelatihan.

Algoritma ini akan digunakan sebagai model dasar dalam proyek ini. Tujuannya sendiri adalah untuk mendapatkan tolok ukur performa paling sederhana. Seberapa baik kita bisa memprediksi popularitas hanya dengan mengasumsikan hubungannya lurus (linier) dengan fitur-fitur audio?

Random Forest Regressor (Model Standar yang Kuat):

Random Forest adalah algoritma machine learning berbasis ensemble learning yang digunakan untuk tugas klasifikasi maupun regresi. Algoritma ini bekerja dengan membangun sekumpulan pohon keputusan (decision trees) selama proses pelatihan, lalu menggabungkan hasil prediksi dari masing-masing pohon untuk menghasilkan keputusan akhir yang lebih akurat dan stabil.

Pada dasarnya, Random Forest merupakan pengembangan dari algoritma Decision Tree, dengan tujuan mengurangi risiko overfitting dan meningkatkan akurasi model. Alih-alih mengandalkan satu pohon keputusan, Random Forest membuat banyak pohon (disebut forest) dan melakukan voting (untuk klasifikasi) atau rata-rata (untuk regresi) dari hasil prediksi semua pohon.

    Peran: Ini adalah model "pekerja keras" kita. Random Forest mampu menangkap hubungan yang lebih rumit (non-linier) antara fitur audio dan popularitas. Biasanya, model ini akan memberikan peningkatan performa yang signifikan dari Linear Regression.

XGBoost (Extreme Gradient Boosting) (Model Performa Tinggi):

XGBoost (Extreme Gradient Boosting) adalah algoritma machine learning berbasis ensemble learning yang mengimplementasikan teknik gradient boosting dengan berbagai optimasi untuk meningkatkan performa dan efisiensi. Algoritma ini dikembangkan oleh Tianqi Chen dan sangat populer karena kemampuannya menghasilkan model dengan akurasi tinggi, waktu pelatihan cepat, serta fleksibilitas tinggi untuk tugas klasifikasi dan regresi.

Algortima ini bekerja dengan cara boosting, yaitu meningkatkan kinerja model secara iteratif:

1. Menghitung kesalahan dari prediksi sebelumnya (residuals).

2. Membangun pohon baru untuk memprediksi kesalahan tersebut.

3. Menggabungkan prediksi semua pohon dengan bobot tertentu untuk menghasilkan output akhir.

    Peran: Ini adalah model "juara" kita. XGBoost seringkali memberikan akurasi tertinggi untuk data tabular seperti ini. Kita akan menggunakannya untuk melihat apakah kita bisa mendapatkan performa terbaik.
- Menganalisis hubungan fitur audio seperti danceability, energy, valence, acousticness, dan lainnya terhadap skor popularitas.
- Menyajikan visualisasi tren audio dari masa ke masa untuk memahami evolusi musik secara kuantitatif.

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari pengguna Kaggle bernama [Yamac Eren Ay](https://www.kaggle.com/yamaerenay) dan pertama kali diunggah pada bulan Januari 2025. Dataset tersebut berjudul "Spotify Dataset 1921-2020, 160k+ Tracks"

Sumber Data: [https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks](Spotify Dataset 1921-2020).

### Explanatory Data Analysis

**1. Distribusi Tiap-Tiap Fitur**
[Distribusi Fitur-Fitur Audio Utama](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/Distribusi-Fitur.png)

Secara keseluruhan, dataset Spotify ini berisi musik yang cenderung:
1. Modern & Non-Akustik: Ditandai dengan energy dan loudness yang tinggi, serta acousticness yang rendah.
2. Berorientasi Vokal: instrumentalness dan speechiness sangat rendah.
3. Cukup Danceable & Beragam Mood: danceability terpusat di tengah-atas, dan valence tersebar merata.
4. Didominasi Lagu Tidak Populer: Sebagian besar data memiliki skor popularity nol.

**2. Hubungan Antar Fitur**
[Hubungan Antar Fitur](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/

Hasil dari diagram ini menunjukan bahwa, faktor kontekstual seperti tahun rilis (year) dan faktor teknis produksi (loudness) ternyata memiliki hubungan yang lebih kuat dengan popularitas daripada karakteristik musikal seperti danceability atau valence. Juga, tidak ada satu pun fitur audio yang memiliki korelasi sangat tinggi (misalnya, di atas 0.8) dengan popularity. Hal ini menegaskan bahwa popularitas lagu adalah fenomena yang kompleks dan tidak bisa ditentukan oleh satu faktor saja.

**3. Evolusi Musik**
[Evolusi Musik](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/Evolusi-Karakteristik-Musik.png)

Dalam diagram ini ditunjukan, bahwa musik populer telah berevolusi dari sesuatu yang akustik, organik, dan lebih dinamis menjadi sesuatu yang keras, energik, digital, dan berorientasi pada ritme. Temuan Hasil ini juga memperkuat temuan sebelumnya dari heatmap, yaitu mengapa year menjadi fitur yang sangat penting. Kolom year secara tidak langsung menjadi rangkuman dari semua tren perubahan drastis ini.

## Data Preparation

Secara garis besar, persiapan data dala  proyek ini terdiri dari 5 tahapan utama:

1.  **Penanganan Data Duplikat dan Kosong:**
    * **Tindakan:** Langkah pertama adalah memeriksa adanya baris data yang duplikat dan data yang kosong (`missing values`) di setiap kolom.
    * **Hasil:** Ditemukan bahwa dataset ini berkualitas sangat baik dan tidak memiliki data kosong sama sekali
      
2.  **Pemilihan Fitur (Feature Selection):**
    * **Tindakan:** Semua kolom yang ada dipilih. Kolom-kolom seperti **`id`**, **`name`** (judul lagu), **`artists`**, dan **`release_date`** telah dihapus.
    * **Alasan:** Kolom-kolom tersebut dihapus karena merupakan *identifier* unik, data teks yang kompleks, atau informasinya sudah terwakili oleh fitur lain (`year`), sehingga tidak cocok untuk langsung digunakan dalam model regresi.

3.  **Pemisahan Fitur (X) dan Target (y):**
    * **Tindakan:** Setelah memilih fitur yang relevan, dataset dibagi menjadi dua bagian:
        * **`X`**: Berisi semua kolom fitur yang akan digunakan sebagai input (seperti year, danceability, energy, dll.).
        * **`y`**: Hanya berisi satu kolom target yang akan diprediksi, yaitu **popularity**.

4.  **Pembagian Dataset (Train-Test Split):**
    * **Tindakan:** Data `X` dan `y` kemudian dibagi menjadi dua set terpisah menggunakan `train_test_split`:
        * **Data Latih (Training Set):** Sebesar 80% dari total data, digunakan untuk "mengajari" model.
        * **Data Uji (Testing Set):** Sebesar 20% dari total data, "disembunyikan" dari model dan hanya digunakan di akhir untuk mengukur performa secara objektif.

5.  **Penskalaan Fitur (Feature Scaling):**
    * **Tindakan:** Merupakan langkah transformasi terakhir. StandardScaler digunakan pada `X_train` dan `X_test`.
    * **Alasan:** Fitur-fitur dalam data memiliki skala yang sangat berbeda (misalnya, duration_ms bernilai ratusan ribu, sementara danceability hanya 0-1). Scaling bertujuan untuk menyamakan skala semua fitur tersebut agar model dapat menilai pengaruh setiap fitur secara adil, tanpa bias terhadap fitur dengan angka yang besar.

## Modeling 






