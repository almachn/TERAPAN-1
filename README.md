# tar deh judulnya besok

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

*Popularitas lagu adalah sesuatu yang kompleks dan dipengaruhi oleh banyak faktor. Namun, dengan adanya ratusan ribu lagu dan puluhan fitur audio, sulit untuk secara manual menentukan pola umum yang membedakan lagu populer dari yang kurang terdengar. Bagaimana kita bisa mengidentifikasi fitur-fitur kunci yang mendorong popularitas suatu lagu, dan bagaimana tren ini berubah dari waktu ke waktu?

### Goal

1. Mengidentifikasi fitur-fitur audio apa yang paling berkontribusi terhadap popularitas lagu.
2. Menganalisis bagaimana tren musik berubah dari tahun ke tahun berdasarkan fitur akustik.
3. Membangun model prediksi popularitas lagu berdasarkan fitur-fiturnya.
4. Memberikan wawasan strategis bagi industri musik berbasis data historis.

### Solution Statements

- Melakukan eksplorasi data terhadap lebih dari 170.000 lagu dari Spotify (1921–2020).
- Menggunakan algoritma machine learning regresi (Linear Regression, Random Forest, Gradient Boosting, XGBoost) untuk memprediksi tingkat popularitas.
- Menganalisis hubungan fitur audio seperti danceability, energy, valence, acousticness, dan lainnya terhadap skor popularitas.
- Menyajikan visualisasi tren audio dari masa ke masa untuk memahami evolusi musik secara kuantitatif.
