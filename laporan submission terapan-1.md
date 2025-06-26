# Analisis Faktor Penentu Popularitas Lagu dan Tren Audionya dari Masa ke Masa (Studi Kasus: Dataset Spotify 1921-2020)

Proyek ini mengeksplorasi dataset lagu Spotify dari tahun 1921 hingga 2020 dengan berbagai fitur akustik, tren popularitas, dan karakteristik audio

## Project Overview
Musik telah menjadi bagian tak terpisahkan dari kehidupan manusia—bukan hanya sebagai bentuk hiburan, tetapi juga sebagai media ekspresi, identitas budaya, bahkan alat politik dan sosial. Dalam satu abad terakhir, industri musik mengalami perubahan yang sangat masif, baik dari sisi gaya, teknologi produksi, hingga cara distribusinya. Dari piringan hitam ke kaset, dari CD ke MP3, hingga kini mendominasi melalui layanan streaming digital seperti Spotify [1](https://santika.upnjatim.ac.id/submissions/index.php/santika/article/view/201/96).

Di era streaming, data adalah raja. Spotify, sebagai salah satu platform musik terbesar di dunia, menyimpan jejak digital dari jutaan lagu dan interaksinya dengan pengguna. Setiap lagu yang diunggah ke platform ini tidak hanya terdiri dari judul dan artisnya saja, tetapi juga diperkaya dengan beragam fitur audio: mulai dari danceability, energy, acousticness, valence, hingga tempo dan liveness. Fitur-fitur ini dihasilkan dari pemrosesan sinyal digital dan menjadi representasi numerik dari karakteristik suara lagu tersebut. Namun, tidak semua lagu mendapatkan popularitas yang sama. Sebagian lagu bisa merajai tangga lagu, viral di media sosial, dan menjadi bagian dari budaya pop; sementara sebagian lainnya nyaris tidak terdengar. 

Lalu pertanyaannya: apa sebenarnya yang membuat sebuah lagu menjadi populer? Apakah ada pola tersembunyi dalam data audio yang bisa kita ungkap? Apakah ada karakteristik tertentu yang menjadi tren pada periode-periode waktu tertentu?

Dengan memanfaatkan dataset Spotify dari tahun 1921 hingga 2020 yang berisi lebih dari 170.000 lagu, kita memiliki peluang langka untuk menggali lebih dalam tentang evolusi musik selama hampir satu abad. Tidak hanya itu, kita juga dapat membangun model prediktif yang dapat membantu memetakan kemungkinan popularitas sebuah lagu berdasarkan fitur-fiturnya.

Studi ini menjadi semakin relevan di tengah era di mana industri musik semakin data-driven. Produser, label, bahkan musisi independen kini sangat terbantu dengan insight berbasis data dalam menentukan gaya produksi, susunan musik, hingga strategi perilisan. Oleh karena itu, penelitian ini tidak hanya memiliki nilai teoritis dalam dunia data science dan musikologi, tetapi juga nilai praktis yang signifikan dalam pengambilan keputusan industri hiburan.

Melalui pendekatan eksploratif dan pemodelan machine learning, kita dapat memberikan gambaran menyeluruh mengenai bagaimana musik populer berubah dari masa ke masa dan apa yang membentuknya. Dengan begitu, kita bisa menjawab pertanyaan penting: “Seperti apa lagu yang disukai dunia?” dan “Apakah kita bisa menciptakan hit berikutnya hanya dari angka-angka?”

## Business Understanding

### Problem Statement

Popularitas lagu adalah sesuatu yang kompleks dan dipengaruhi oleh banyak faktor. Namun, dengan adanya ratusan ribu lagu dan puluhan fitur audio, sulit untuk secara manual menentukan pola umum yang membedakan lagu populer dari yang kurang terdengar. Bagaimana kita bisa mengidentifikasi fitur-fitur kunci yang mendorong popularitas suatu lagu, dan bagaimana tren ini berubah dari waktu ke waktu?

### Goal

1. Mengidentifikasi fitur-fitur apa yang paling berkontribusi terhadap popularitas lagu.
2. Menganalisis bagaimana tren musik berubah dari tahun ke tahun berdasarkan fitur akustik.
3. Membangun model prediksi popularitas lagu berdasarkan fitur-fiturnya.
4. Memberikan wawasan strategis bagi industri musik berbasis data historis.

### Solution Statements

Untuk mencapai tujuan utama, yakni memprediksi popularits lagu berdasarkan karakteristik audio dan metadata, beberapa pendekatan (algoritma prediktif dan teknik analisis fitur) telah diimplementasikan dalam proyek ini:

**1. Linear Regression**

- **Pendekatan:** Model regresi linier digunakan sebagai baseline sederhana yang mengasumsikan hubungan linier antara fitur dan target popularitas. [2](https://doi.org/10.32877/bt.v7i3.2228)

- Teknik: Menggunakan LinearRegression dari sklearn untuk mempelajari hubungan antara fitur numerik seperti year, acousticness, danceability, dll., terhadap skor popularitas. Koefisien model dianalisis untuk mengukur besarnya pengaruh masing-masing fitur (dalam bentuk nilai absolut).

**2. Random Forest Regressor:**

- **Pendekatan:** Algoritma ensemble berbasis decision tree yang mampu menangani hubungan non-linear serta interaksi kompleks antar fitur. [3](https://doi.org/10.35970/infotekmesin.v14i1.1751).

- **Teknik:** Menggunakan Random Forest Regressor dari sklearn.ensemble untuk membuat banyak decision tree dan menggabungkan prediksinya. Model ini juga memberikan fitur feature_importances_ untuk mengevaluasi pengaruh relatif setiap fitur terhadap prediksi.

**3. XGBoost Regressor**

- **Pendekatan:** Model gradient boosting yang kuat dan efisien, dirancang untuk performa tinggi dalam berbagai jenis data. [4](https://doi.org/10.36040/jati.v7i5.7308).

- **Teknik:** Menggunakan XGBRegressor dari library xgboost, yang membangun model secara bertahap dengan mengoreksi kesalahan dari prediksi sebelumnya. Fitur penting diambil berdasarkan bobot kontribusi setiap fitur dalam proses boosting.

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari pengguna Kaggle bernama **[Yamac Eren Ay](https://www.kaggle.com/yamaerenay)** dan pertama kali diunggah pada bulan Januari 2025. Dataset tersebut berjudul "Spotify Dataset 1921-2020, 160k+ Tracks"

**Sumber Data:** [Spotify Dataset 1921-2020](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks)

**Karakteristik Dataset Asli (Spotify Dataset 1921-2020 160K Tracks):**

- **Jumlah Kolom:** 19 Kolom (valence, year, acousticness, artists, danceability. duration_ms, energy, explicit, id, instrumentalness, key, liveness, loudness, mode, name, popularity, release_date, speechiness, tempo)
- **Jumlah Entries:** 170,653 Entries pada Tiap-Tiap Kolom
- **Tipe Data:** 9 Kolom Float, 6 Kolom Integer, dan 4 Kolom Object

**Kondisi Data:** 

Sebelum masuk ke tahap data preparation, dilakukan peninjauan awal terhadap kondisi dataset guna mengevaluasi kualitas data serta mengidentifikasi potensi permasalahan. Pemeriksaan ini mencakup beberapa aspek penting, antara lain:
   1. **Struktur dan Tipe Data:** Melalui fungsi info(), diketahui bahwa seluruh kolom dalam subset dataset tidak mengandung nilai kosong (null).
   2. **Nilai yang Hilang:** Meskipun tidak ditemukan nilai kosong pada kolom-kolom utama, fungsi dropna() tetap digunakan sebagai langkah preventif untuk memastikan kebersihan data saat pengolahan lebih lanjut dilakukan pada dataset lengkap.
   3. **Data Duplikat:** Tidak terdapat baris yang duplikat yang ditemukan. Meski demikian, drop_duplicates() tetap dijalankan untuk menjaga konsistensi data secara menyeluruh.

### Explanatory Data Analysis

**1. Informasi Dataset**

Statistik Dasar Data [1]

![Statistik Dasar 1](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/dasar-1.png)

Statistik Dasar Data [2]

![Statistik Dasar 2](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/dasar-2.png)

**2. Distribusi Tiap-Tiap Fitur**

![Distribusi Fitur-Fitur Audio Utama](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/Distribusi-Fitur.png)

Secara keseluruhan, dataset Spotify ini berisi musik yang cenderung:

1. Modern & Non-Akustik: Ditandai dengan energy dan loudness yang tinggi, serta acousticness yang rendah.
2. Berorientasi Vokal: instrumentalness dan speechiness sangat rendah.
3. Cukup Danceable & Beragam Mood: danceability terpusat di tengah-atas, dan valence tersebar merata.
4. Didominasi Lagu Tidak Populer: Sebagian besar data memiliki skor popularity nol.

**3. Hubungan Antar Fitur**

![Hubungan Antar Fitur](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/111.png)

Hasil dari diagram ini menunjukan bahwa, faktor kontekstual seperti tahun rilis (year) dan faktor teknis produksi (loudness) ternyata memiliki hubungan yang lebih kuat dengan popularitas daripada karakteristik musikal seperti danceability atau valence. Juga, tidak ada satu pun fitur audio yang memiliki korelasi sangat tinggi (misalnya, di atas 0.8) dengan popularity. Hal ini menegaskan bahwa popularitas lagu adalah fenomena yang kompleks dan tidak bisa ditentukan oleh satu faktor saja.

**4. Evolusi Musik**

![Evolusi Musik](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/Evolusi-Karakteristik-Musik.png)

Dalam diagram ini ditunjukan, bahwa musik populer telah berevolusi dari sesuatu yang akustik, organik, dan lebih dinamis menjadi sesuatu yang keras, energik, digital, dan berorientasi pada ritme. Temuan Hasil ini juga memperkuat temuan sebelumnya dari heatmap, yaitu mengapa year menjadi fitur yang sangat penting. Kolom year secara tidak langsung menjadi rangkuman dari semua tren perubahan drastis ini.

## Data Preparation

Secara garis besar, persiapan data dala  proyek ini terdiri dari 5 tahapan utama:

1.  **Penanganan Data Duplikat dan Kosong:**
    * **Tindakan:** Langkah pertama adalah memeriksa adanya baris data yang duplikat dan data yang kosong (`missing values`) di setiap kolom.
    * **Hasil:** Ditemukan bahwa dataset ini berkualitas sangat baik dan tidak memiliki data kosong sama sekali
      
2.  **Pemilihan Fitur (Feature Selection):**
    * **Tindakan:** Semua kolom yang ada dipilih. Kolom-kolom seperti **id**, **name** (judul lagu), **artists**, dan **release_date** telah dihapus.
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

Setelah melalui tahap persiapan data, data yang telah siap digunakan akan dimanfaatkan untuk membangun model. Pada tahap ini, akan dikembangkan tiga model yang berbeda sebagai bahan perbandingan

### Pembuatan model menggunakan algoritma Linear Regression

*Algoritma ini akan digunakan sebagai model dasar dalam proyek ini. Tujuannya sendiri adalah untuk mendapatkan tolok ukur performa paling sederhana. Seberapa baik kita bisa memprediksi popularitas hanya dengan mengasumsikan hubungannya lurus (linier) dengan fitur-fitur audio?*

- **Cara Kerja:** Linear Regression mencoba membentuk garis lurus terbaik yang bisa menghubungkan fitur input (X) dengan target output (y). Model ini berusaha mencari koefisien (β) untuk setiap fitur agar selisih antara prediksi dan nilai asli (error) sekecil mungkin.

- **Parameter Penting:**
  
  1. fit_intercept: Apakah model harus menghitung intercept (bias/konstanta) atau tidak.
  2. normalize / standardize: Untuk normalisasi fitur (sudah deprecated di versi baru scikit-learn).
  3. n_jobs: Jumlah core CPU untuk paralel
    
- **Kelebihan dan Kekurangannya:**
  
  **Kelebihan:**
  1. Cepat dan Efisien: Linear Regression sangat ringan secara komputasi, sehingga cocok sebagai model baseline untuk menguji apakah fitur-fitur seperti year, acousticness, dll. yang mempunyai hubungan linier dengan popularity.
  2. Interpretatif: Nilai koefisiennya dapat langsung menunjukkan arah dan kekuatan pengaruh setiap fitur. Misalnya, koefisien besar pada year, maka langsung terlihat bahwa semakin baru lagu, semakin populer.

   **Kekurangan:**
  1. Terlalu simpel untuk fenomena kompleks: Popularitas lagu bukan cuma dipengaruhi satu-dua fitur dengan hubungan lurus. Misalnya, pengaruh energy atau danceability mungkin baru terasa jika dikombinasikan dengan fitur lain — hal ini tidak bisa ditangkap oleh model linier.
  2. Rentan underfitting: Karena model ini tidak bisa menangkap pola non-linear atau interaksi fitur, hasilnya bisa terlalu sederhana dan tidak mencerminkan kenyataan yang kompleks di dunia musik streaming.

### Pembuatan model mengguankan algoritma Random Forest

*Model ini diasumsikan sebagai menjadi model "pekerja keras" karena kemampuannya menangkap hubungan yang lebih rumit (non-linier) antara fitur audio dan popularitas. Biasanya, model ini akan memberikan peningkatan performa yang signifikan dari Linear Regression.*

- **Cara Kerja:** Random Forest adalah ensemble dari banyak decision tree. Ia membentuk banyak pohon keputusan (tree) dari subsample data + fitur acak, lalu merata-ratakan hasilnya untuk regresi. Pendekatan ini membantu mengurangi overfitting yang umum pada decision tree tunggal.

- **Parameter Penting:**

  1. n_estimators: Jumlah pohon yang dibangun.
  2. max_depth: Kedalaman maksimal pohon. Lebih dalam = lebih kompleks.
  3. min_samples_split: Minimum jumlah sampel untuk membagi node.
  4. max_features: Berapa banyak fitur yang dipertimbangkan per split.
  5. random_state: Supaya hasil replikasi tetap sama.
     
- **Kelebihan dan Kekurangannya:**
  
  **Kelebihan:**
  1. Menangkap pola non-linear dan interaksi fitur: Random Forest mampu mengenali bahwa misalnya lagu yang akustik tapi energik punya popularitas tinggi di tahun tertentu. Hal ini membuatnya jauh lebih fleksibel dibanding Linear Regression.
  2. Tahan terhadap outlier: Popularitas lagu yang ekstrem (sangat tinggi atau sangat rendah) tidak akan mengganggu model terlalu parah.
  3. Memberikan informasi feature importance: Sangat berguna untuk mengetahui fitur apa yang secara keseluruhan paling banyak digunakan oleh model saat memutuskan prediksi. Misalnya, loudness, year, dan danceability.

   **Kekurangan:**
  1. Kurang transparan: Meskipun feature importance bisa diketahui, tetapi bagaimana kombinasi fitur menghasilkan nilai prediksi tertentu tidak bisa diketahui secara persis. Hal ini membuat hasilnya agak sulit untuk dijelaskan ke non-teknikal stakeholder.
  2. Model besar dan berat: Untuk dataset besar seperti Spotify (170k+ lagu), proses training bisa lumayan memakan waktu dan RAM, apalagi jika tidak dibatasi max_depth atau jumlah pohon (n_estimators).

### Pembuatan model menggunakan algoritma XGBoost

*XGBoost seringkali memberikan akurasi tertinggi untuk data tabular seperti ini. Model ini akan digunakan untuk melihat apakah hasil uji bisa mendapatkan performa terbaik.*

- **Cara Kerja:** XGBoost (Extreme Gradient Boosting) adalah model boosting yang membangun pohon berurutan. Setiap pohon baru belajar dari kesalahan pohon sebelumnya (sisa error yang belum ditebak dengan baik). Dengan teknik boosting ini, model jadi makin presisi seiring pohon ditambahkan.

- **Parameter Penting:**

  1. n_estimators: Jumlah total boosting round.
  2. learning_rate: Seberapa besar kontribusi tiap pohon baru → kecil = pelan tapi akurat.
  3. subsample: Persentase data yang digunakan untuk tiap pohon.
  4. colsample_bytree: Fitur yang digunakan per pohon.
  5. reg_alpha, reg_lambda: Regularisasi untuk mencegah overfitting.
     
- **Kelebihan dan Kekurangannya:**
  
  **Kelebihan:**
  1. Presisi tinggi: XGBoost sangat powerful dalam mengejar akurasi prediksi. Cocok digunakan dalam proyek ini yang memiliki tujuan membuat model prediksi popularitas dengan performa optimal.
  2. Menangani kompleksitas dengan 'elegan': Model ini dapat menangkap pengaruh fitur yang hanya muncul di kondisi tertentu (contoh: lagu akustik cenderung tidak populer kecuali jika rilis di tahun 2010 ke atas).
  3. Mempunyai regularisasi bawaan: Membantu mengurangi risiko overfitting — cocok untuk dataset besar dengan banyak fitur seperti ini.

   **Kekurangan:**
  1. Butuh tuning yang teliti: Untuk hasil maksimal, parameter seperti learning_rate, max_depth, subsample, dll., harus disetel dengan hati-hati. Tanpa itu, model bisa underperform atau overfit.
  2. Kurang interpretatif: Sama seperti Random Forest, model ini tidak bisa dengan mudah bilang "fitur A menaikkan popularitas sebesar sekian". Cocok untuk model akhir, tapi bukan untuk insight yang mudah dijelaskan.

## Evaluasi 

**Proses evaluasi pada model ini menggunakan RMSE (Root Mean Squared Error) dan R-Squared (R2 Score)**

Formula RMSE:

![Formula RMSE](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/formulaRMSE.png)

**Cara Kerja:** RMSE mengukur seberapa jauh prediksi model meleset dari nilai sebenarnya, dihitung dengan cara:

1. Ambil selisih antara nilai prediksi dan nilai asli (y_pred - y_true).
2. Kuadratkan semua selisih tadi (biar nggak ada nilai negatif).
3. Hitung rata-rata dari semua nilai kuadrat tersebut (Mean Squared Error / MSE).
4. Ambil akar kuadrat dari MSE → itulah RMSE.

Formula R-Squared:

![Formula R2](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/formulaR2.png)

**Cara Kerja:** R² mengukur seberapa baik model menjelaskan variasi data target. Skor ini membandingkan performa model dengan baseline paling sederhana: rata-rata semua nilai popularity.

1. R² = 1 → prediksi model sempurna.
2. R² = 0 → model nggak lebih baik dari sekadar menebak rata-rata.
3. R² < 0 → model lebih buruk dari rata-rata.

### Hasil Evaluasi

**1. Linear Regression**

![Evaluasi Model Linear Regression](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/LR.png)

Pada gambar di atas, dapat dilihat bahwa saat proses pengujian, hasil prediksi tidak akurat dengan nilai sebenarnya.
Secara RMSE, diketahui bahwa rata-rata prediksi skor popularitas yang dibuat oleh model meleset sekitar 10.7 poin dari skor popularitas yang sebenarnya (pada skala 0-100).
Secara R2 Squared, diketahui bahwa model mampu menjelaskan sekitar 75.9% dari keragaman data yang memengaruhi popularitas sebuah lagu berdasarkan fitur-fitur yang diberikan. Sisa ~24% lainnya dipengaruhi oleh faktor-faktor yang tidak bisa ditangkap oleh model ini. 

**2. Hasil Evaluasi Model Random Forest**

![Evaluasi Model Random Forest](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/RF.png)

Berdasarkan gambar di atas, algoritma Random Forest menunjukkan performa yang lebih baik dibandingkan LinearRegression, ditandai dengan nilai hasil Root Mean Squared Error (RMSE) yang lebih rendah, yakni sebesar 9.5558 dan R Squared (R2 Score) sebesar 0.8092. Hasil ini menunjukan bahwa Random Forest lebih efektif dalam melakukan prediksi pada dataset yang digunakan.

**3. Hasil Evaluasi Model XGBoost**

![Evaluasi Model XGBoost](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/XGBoost.png)

Model XGboost memberikan nilai hasil evaluasi terbaik, dengan besaran RMSE 9.5049 dan R2-Squared 0.8112. 

Berikut merupakan perbandingan hasil evaluasi ketiga model
![Evaluasi Model Linear Regression](https://raw.githubusercontent.com/almachn/TERAPAN-1/main/assets/perbandingan-final.png)

## Kesimpulan

Berdasarkan laporan proyek machine learning yang telah disusun, berikut adalah kesimpulan utama yang merangkum pemecahan problem statement, pencapaian pendekatan solusi, hasil evaluasi model, serta keterbatasan proyek:

### 1. Identifikasi Masalah dan Tujuan yang Tercapai

- Proyek ini berhasil mengidentifikasi permasalahan utama dalam ekosistem musik digital: bagaimana memprediksi popularitas lagu berdasarkan karakteristik audio dan metadata.
- Tujuan utama untuk membangun model prediksi yang mampu mengenali pola dalam fitur-fitur lagu dan menghasilkan estimasi popularitas yang akurat telah tercapai, melalui penerapan berbagai algoritma regresi.

### 2. Implementasi Metode dan Pendekatan Solusi

- Linear Regression digunakan sebagai baseline untuk menangkap hubungan linier antar fitur dan memberikan interpretasi awal atas pengaruh variabel seperti year, acousticness, dan energy.
- Random Forest Regressor diimplementasikan untuk menangani hubungan non-linear dan mengevaluasi pentingnya interaksi fitur teknis audio seperti loudness dan danceability.
- XGBoost Regressor dikembangkan sebagai pendekatan boosting yang menggabungkan presisi tinggi dengan kemampuan generalisasi, dan berhasil menunjukkan performa terbaik dalam hal evaluasi.

Seluruh pendekatan telah berhasil diimplementasikan dan dikaji melalui proses pemodelan, training, evaluasi, dan analisis feature importance secara visual.

### 3. Hasil Evaluasi Kinerja dan Kaitannya dengan Problem Statement

- Evaluasi menggunakan metrik RMSE dan R2 menunjukkan bahwa model mampu menjelaskan hingga 81% variasi dalam skor popularitas lagu, dengan kesalahan prediksi rata-rata berkisar 6 poin.
- Model XGBoost memiliki performa tertinggi, diikuti oleh Random Forest dan Linear Regression. Hal ini menunjukkan bahwa popularitas lagu merupakan fenomena kompleks yang lebih baik dipahami oleh model yang menangkap pola non-linear dan interaksi fitur.
- **Hasil evaluasi memperkuat keyakinan bahwa karakteristik waktu (year) dan teknis audio (loudness, acousticness, energy) adalah penentu utama popularitas, menjawab problem statement terkait pemahaman tren popularitas musik.**

### 4. Insight, Kegunaan, dan Relevansi Solusi

- Model dapat digunakan untuk:
  1. Memprediksi popularitas lagu-lagu baru sebelum diliris di publik
  2. Membantu musisi, label, atau platform streaming dalam mengindentifikasi atribut audo yang cenderung disukai penggemar.
- Analisis feature importance juga memberikan insight mendalam mengenai elemen-elemen produksi audio yang berkontribusi terhadap kesuksesan sebuah lagu di platform seperti Spotify.
- Solusi ini berpotensi mendukung proses kurasi otomatis, pemasaran berbasis data, dan pengembangan konten audio yang lebih terpersonalisasi.

### Keterbatasan Proyek

- Dataset yang digunakan meskipun cukup besar, tidak mempertimbangkan faktor eksternal seperti promosi, artis, atau algoritma distribusi Spotify, yang juga mempengaruhi popularitas lagu.
- Model regresi yang diterapkan berfokus pada fitur numerik dan teknis, tanpa mempertimbangkan lirik, genre, atau emosi lagu yang bisa menambah konteks prediktif.
- Proyek ini tidak mencakup evaluasi temporal (time-series) untuk memprediksi perubahan popularitas dari waktu ke waktu


Secara keseluruhan, proyek ini telah berhasil mengimplementasikan berbagai algoritma machine learning untuk memprediksi popularitas lagu berdasarkan fitur audio dan metadata.

Model yang dikembangkan mampu mengidentifikasi pola yang signifikan, memberikan insight praktis yang dapat dimanfaatkan oleh industri musik digital, dan menunjukkan bahwa popularitas adalah gabungan kompleks antara waktu rilis dan kualitas teknis produksi.

Meskipun terdapat ruang untuk pengembangan lebih lanjut, seperti integrasi data eksternal atau fitur lirik, proyek ini memberikan fondasi kuat bagi upaya data-driven decision making dalam dunia musik dan konten digital.











[1]: https://santika.upnjatim.ac.id/submissions/index.php/santika/article/view/201/96

[2]: https://doi.org/10.32877/bt.v7i3.2228

[3]: https://doi.org/10.35970/infotekmesin.v14i1.1751

[4]: https://doi.org/10.36040/jati.v7i5.7308

[5]: https://doi.org/10.29207/resti.v5i1.2815






