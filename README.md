# tar deh judulnya besok

## Domain Proyek 
Domain utama proyek ini berfokus pada Bidang Ekonomi dan Bisnis, khususnya Real Estat. Fokus utamanya berupa pembangunan model untuk memahami dan memprediksi valuasi properti residensial. Model ini kemudian akan diperkaya dengan analisis dari domain Keuangan & Ekonomi, di mana wawasan yang dihasilkan akan diinterpretasikan dalam konteks kondisi makroekonomi saat ini, seperti tekanan inflasi, untuk memberikan wawasan strategis.

### Latar Belakang
Dalam kehidupan, setiap manusia memiliki kebutuhan dasar yang harus dipenuhi demi kelangsungan hidup yang layak. Kebutuhan primer—seperti sandang (pakaian), pangan (makanan), dan papan (tempat tinggal)—menjadi fondasi utama dalam menjalani kehidupan sehari-hari. Di antara ketiga kebutuhan tersebut, papan atau tempat tinggal memiliki peranan yang sangat penting. Rumah bukan hanya sekadar tempat berlindung dari panas dan hujan, melainkan juga menjadi ruang personal untuk beristirahat, tumbuh, dan membangun kehidupan bersama keluarga. [1](https://doi.org/10.35794/emba.v11i3.50231)

Namun, seiring dengan perkembangan zaman dan meningkatnya taraf hidup masyarakat, rumah tidak lagi hanya dipandang dari sisi fungsionalnya saja. Kini, rumah juga menjadi simbol status sosial dan ukuran pencapaian hidup seseorang. Kepemilikan rumah yang besar, mewah, dan berlokasi strategis sering kali dianggap sebagai representasi kesuksesan. Fenomena ini didorong oleh faktor sosial seperti gengsi dan keinginan untuk diakui atau dipandang lebih tinggi oleh lingkungan sekitar. Tak bisa dimungkiri, dalam masyarakat modern, citra dan penilaian terhadap seseorang kerap kali dibentuk oleh apa yang mereka miliki—termasuk rumah yang mereka huni. [2](https://doi.org/10.36040/jati.v7i1.6343).

Dalam proyek ini, akan dibuat beberapa model Machine Learning yang kemudian dievaluasi untuk membandingkan model mana yang hasil prediksinya paling baik. Model yang unggul tersebut diharapkan dapat diandalkan untuk memprediksi kisaran harga rumah yang sesuai berdasarkan kondisi dan karakteristik spesifik dari properti yang dinilai.

## Business Understanding

### Problem Statement

Berdasarkan proses klarifikasi di atas, berikut adalah pernyataan masalah utama yang akan dijawab dalam proyek ini:

1. Valuasi properti saat ini seringkali bersifat subjektif, tidak konsisten, dan lambat, sehingga menciptakan ketidakpastian dan risiko finansial yang signifikan bagi para pemangku kepentingan.
2. Para pemangku kepentingan tidak memiliki cara yang mudah untuk memahami secara kuantitatif faktor-faktor atau atribut properti apa saja yang paling signifikan dalam menentukan harga sebuah rumah.
3. Ketiadaan sebuah model acuan yang transparan dan berbasis data menyebabkan terjadinya inefisiensi di pasar, seperti potensi overpricing oleh penjual atau underpaying oleh pembeli.

### Goal
Untuk mengatasi pernyataan masalah tersebut, tujuan utama dari proyek ini adalah:

1. Mengembangkan sebuah model acuan yang mampu memberikan estimasi harga rumah yang objektif untuk meningkatkan transparansi dan mengurangi subjektivitas dalam proses valuasi.
2. Mengidentifikasi dan memeringkat faktor-faktor penentu harga rumah yang paling berpengaruh, sehingga memberikan wawasan yang dapat ditindaklanjuti (actionable insights) bagi pembeli, penjual, dan investor.
3. Menyediakan solusi berbasis data yang dapat membantu mengurangi risiko finansial dan meningkatkan efisiensi pasar properti secara keseluruhan.

### Solution Statements

Untuk mencapai tujuan-tujuan di atas, berikut adalah solusi terukur yang diajukan:

- Pembangunan dan Analisis Komparatif Model Regresi
Membangun dan melatih setidaknya dua algoritma regresi yang berbeda, yaitu Linear Regression sebagai model dasar (baseline) dan Random Forest atau Gradient Boosting sebagai model yang lebih kompleks. Tujuannya adalah untuk membandingkan performa dari berbagai tingkat kompleksitas model.

- Pengukuran Keberhasilan: Solusi ini akan dievaluasi dengan membandingkan metrik performa dari setiap model pada data uji. Secara spesifik, akan digunakan Root Mean Squared Error (RMSE), di mana nilai yang lebih rendah menandakan prediksi yang lebih akurat. Selain itu, akan digunakan R-squared (R2) untuk melihat seberapa besar variasi harga yang dapat dijelaskan oleh model. Model yang memiliki RMSE terendah dan R2 tertinggi akan dianggap sebagai arsitektur terbaik.

