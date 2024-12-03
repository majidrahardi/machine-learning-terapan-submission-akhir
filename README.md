# Laporan Proyek Akhir Machine Learning Terapan - Majid Rahardi

## Domain Proyek
Dalam beberapa tahun terakhir, anime telah berkembang menjadi bentuk hiburan global yang sangat populer, dengan jutaan penggemar di seluruh dunia. Seiring dengan meningkatnya jumlah judul anime yang tersedia di berbagai platform streaming seperti Crunchyroll, Netflix, dan Funimation, penggemar anime sering kali dihadapkan dengan pilihan yang sangat banyak. Hal ini membuat mereka kesulitan untuk menemukan anime yang sesuai dengan preferensi pribadi mereka. Di sinilah peran sistem rekomendasi menjadi sangat penting [[1](http://jurnal.utu.ac.id/JTI/article/download/7787/4290)].

Sistem rekomendasi adalah alat yang menggunakan algoritma untuk membantu pengguna menemukan konten yang sesuai dengan minat atau preferensi mereka berdasarkan data yang ada. Dalam konteks anime, sistem rekomendasi bertujuan untuk menyarankan anime yang serupa dengan anime yang sudah ditonton atau yang dianggap disukai oleh pengguna, berdasarkan berbagai faktor seperti genre, tema, karakter, dan rating. Sistem ini berfungsi untuk meningkatkan pengalaman pengguna dengan menyederhanakan proses pencarian konten yang relevan dan memperkenalkan mereka pada judul anime baru yang mungkin belum mereka temui sebelumnya [[2](http://repository.uin-malang.ac.id/18842/1/18842.pdf)].

Salah satu teknik yang sering digunakan dalam sistem rekomendasi adalah Content-Based Filtering. Dalam pendekatan ini, algoritma menganalisis konten yang ada dalam database, seperti genre, deskripsi, atau karakteristik lain dari anime, dan membandingkannya dengan anime yang telah ditonton atau disukai oleh pengguna. Salah satu metode untuk mengukur kesamaan konten adalah TF-IDF (Term Frequency - Inverse Document Frequency), yang membantu mengukur seberapa penting suatu kata dalam dokumen (dalam hal ini, genre atau deskripsi anime). Selain itu, Cosine Similarity digunakan untuk menghitung seberapa mirip dua anime berdasarkan vektor TF-IDF mereka, yang memungkinkan sistem untuk memberikan rekomendasi yang relevan [[3](http://repository.uin-malang.ac.id/17878/2/17878.pdf)].

Secara keseluruhan, tujuan dari sistem rekomendasi anime adalah untuk memberikan pengalaman yang lebih personal dan efisien bagi pengguna dalam menemukan anime yang sesuai dengan selera mereka, yang pada akhirnya dapat meningkatkan kepuasan dan keterlibatan mereka dengan platform streaming anime.

## Business Understanding
Dalam industri hiburan digital, terutama pada platform streaming anime, pemirsa menghadapi tantangan besar untuk menemukan konten yang relevan di tengah banyaknya pilihan yang tersedia. Platform-platform besar seperti Crunchyroll, Netflix, dan Funimation memiliki ribuan judul anime yang terus berkembang setiap tahun. Namun, dengan semakin banyaknya pilihan yang ada, pengguna sering kali merasa kesulitan untuk menemukan anime yang sesuai dengan minat pribadi mereka. Masalah utama yang muncul adalah kesenjangan antara preferensi pengguna dengan informasi yang tersedia tentang konten anime, yang membuat proses pencarian menjadi memakan waktu dan kurang efisien.

### Problem Statements
1. **Overload Pilihan**  
   Platform streaming anime memiliki katalog yang sangat besar, yang dapat membuat pengguna kewalahan dalam memilih anime yang sesuai dengan minat mereka.
   
2. **Keterbatasan Rekomendasi yang Tepat**  
   Tanpa adanya sistem yang baik, rekomendasi yang diberikan seringkali tidak relevan dengan selera individu pengguna. Hal ini menyebabkan penurunan kepuasan pengguna dan meningkatkan kemungkinan mereka untuk meninggalkan platform.

### Goals
1. **Meningkatkan Pengalaman Pengguna**  
   Mengembangkan sistem rekomendasi yang dapat memberikan saran anime yang lebih relevan berdasarkan preferensi pribadi pengguna, sehingga meningkatkan kepuasan pengguna.
   
### Solution Statements
1. **Pengumpulan Data dan Pemrosesan Data**  
   Mengumpulkan dan memproses data terkait anime, seperti genre, nama, rating, dan jumlah anggota yang terdaftar untuk setiap anime. Data ini akan digunakan untuk membangun model rekomendasi berbasis konten.
   
2. **Pengembangan Model**  
   Penggunaan TF-IDF Vectorizer untuk mengubah kolom genre atau deskripsi anime menjadi representasi numerik. Metode ini akan mengukur frekuensi kata dan pentingnya kata-kata dalam konteks anime tertentu, serta mengidentifikasi kesamaan antar anime berdasarkan kata-kata yang paling sering muncul. Serta penggunaan Cosine Similarity untuk mengukur tingkat kesamaan antara anime. Anime dengan nilai kesamaan yang lebih tinggi akan dianggap lebih relevan untuk direkomendasikan kepada pengguna.
   
3. **Memberikan Rekomendasi**  
   Berdasarkan hasil perhitungan kesamaan, sistem akan merekomendasikan anime yang memiliki kesamaan konten yang tinggi dengan anime yang sudah disukai atau ditonton oleh pengguna.

4. **Evaluasi Model**
  Mengevaluasi model dengan menghitung nilai presisi dari rekomendasi yang diberikan.

## Data Understanding
### Deskripsi Dataset
Dataset **Anime Recommendations Database** yang diambil dari Kaggle ini berisi informasi mengenai lebih dari 12.000 anime yang tersedia di berbagai platform streaming. Data ini memberikan gambaran yang komprehensif tentang berbagai anime, mencakup berbagai atribut penting seperti **ID anime**, **nama anime**, **genre**, **tipe (movie, TV, OVA, dll.)**, **jumlah episode**, **rating**, dan **jumlah anggota** yang mengikuti anime tersebut. Adapun dataset dapat diakses secara publik di [link dataset](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).

### Ukuran Dataset
- **Jumlah entri (baris)**: 12.294
- **Jumlah atribut (kolom)**: 7

### Atribut dalam Dataset
- **anime_id**: ID unik untuk setiap anime dalam dataset.
- **name**: Nama anime yang bersangkutan.
- **genre**: Genre atau tema utama anime, yang dapat berupa kombinasi genre (misalnya, "Action, Adventure, Fantasy").
- **type**: Tipe anime, seperti TV, Movie, OVA (Original Video Animation), dan sebagainya.
- **episodes**: Jumlah episode yang tersedia dalam anime (untuk tipe TV dan OVA).
- **rating**: Rating yang diberikan oleh pengguna, biasanya diukur dalam skala 1 hingga 10, mencerminkan popularitas atau kualitas anime berdasarkan umpan balik pengguna.
- **members**: Jumlah anggota (pengguna) yang terdaftar atau mengikuti anime tersebut di platform MyAnimeList, yang mencerminkan seberapa populer anime tersebut di kalangan pengguna.

### Eksplorasi Data Awal
- ```python
  df.shape
  ```
  Kode tersebut memiliki luaran:
  ```python
  (12294, 7)
  ```
- Jumlah entri (baris): 12.294
- Jumlah atribut (kolom): 7


- ```python
  df.keys()
  ```
  Kode tersebut memiliki luaran:
  ```python
  Index(['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'], dtype='object')
  ```
Melihat Nama variabel pada dataset

### Cek Missing Value 
- ```python
  df.isnull().sum()
  ```
  Kode tersebut memiliki luaran:

![CekMissingValue](https://github.com/user-attachments/assets/f3b843e5-8269-4f4d-8880-56c4656c5f1d)


Output dari perintah `df.isnull().sum()` menunjukkan jumlah nilai yang hilang (null) pada setiap kolom dalam dataset. Kolom **`anime_id`**, **`name`**, **`episodes`**, dan **`members`** tidak memiliki nilai yang hilang, masing-masing menunjukkan 0 nilai null. Namun, kolom **`genre`** memiliki 62 nilai null, yang menunjukkan bahwa ada 62 anime yang tidak memiliki informasi genre. Kolom **`type`** memiliki 25 nilai null, yang berarti 25 anime tidak memiliki tipe yang terdefinisi. Terakhir, kolom **`rating`** memiliki 230 nilai null, yang menunjukkan bahwa ada 230 anime yang tidak memiliki rating yang diberikan oleh pengguna. Nilai null ini perlu ditangani sebelum analisis lebih lanjut, misalnya dengan mengisi nilai yang hilang atau menghapus baris yang terkait.

### Cek Duplicated Data 
- ```python
  print(duplicated_data)
  ```
  Kode tersebut memiliki luaran:
  
![CekDuplicated](https://github.com/user-attachments/assets/6033d94f-6b6c-4093-9723-55a7ac2c4f45)

Output tersebut menunjukkan bahwa tidak ada data duplikat dalam dataset, yang berarti setiap catatan atau baris dalam DataFrame adalah unik. Hal ini penting karena memastikan bahwa analisis yang dilakukan tidak akan terpengaruh oleh data yang berulang, yang dapat menyebabkan bias atau kesalahan interpretasi.

## Data Preparation
Pada tahap **Data Preparation**, dataset yang telah diunduh dan dimuat akan diproses agar siap digunakan untuk analisis lebih lanjut atau pembangunan model sistem rekomendasi. Proses ini melibatkan serangkaian langkah untuk memastikan bahwa data yang digunakan bersih, konsisten, dan dalam format yang dapat diproses dengan baik oleh model machine learning. Beberapa langkah utama yang dilakukan pada tahap ini antara lain:

### Mengatasi Missing Value
Pada tahap *Mengatasi Missing Value*, kita menangani nilai yang hilang dalam dataset yang dapat mempengaruhi kualitas analisis dan model rekomendasi. Salah satu pendekatan yang umum digunakan adalah dengan menghapus baris yang memiliki missing values, seperti yang dilakukan dengan fungsi `dropna()`, yang dapat mengurangi dampak dari data yang hilang. 

- ```python
	#Cleaning missing value with function dropna()
	df_clean = df.dropna()
	df_clean
  ```
  Kode tersebut memiliki luaran:

![DataClean](https://github.com/user-attachments/assets/82bcad6e-23fd-400c-b3f5-8658841145cd)

  
- ```python
	df_clean.isnull().sum()
  ```
  Kode tersebut memiliki luaran:
  
![CekDataClean](https://github.com/user-attachments/assets/77847e7e-d53a-4469-97de-0122674cc6ee)

Output di atas menunjukkan hasil dari proses pembersihan data dengan menggunakan fungsi dropna() untuk menghapus baris yang memiliki nilai kosong (missing values). Dataset yang dihasilkan, df_clean, kini berisi hanya baris-baris lengkap, tanpa adanya nilai yang hilang pada kolom manapun. Ini dapat membantu dalam analisis dan pemodelan yang memerlukan data yang bersih dan konsisten. Namun, proses ini juga mengakibatkan penghapusan beberapa baris, terutama yang memiliki nilai kosong pada kolom seperti genre, type, dan rating, yang dapat menyebabkan hilangnya informasi penting terkait beberapa anime. Meskipun demikian, dataset yang telah dibersihkan ini lebih mudah dikelola dan digunakan dalam model pembelajaran mesin atau analisis lanjutan, meski harus diingat bahwa penggunaan dropna() mengorbankan data yang tidak lengkap.

## Model Development dengan Content Based Filtering
### TF-IDF Vectorizer

- ```python
	print(tfidf_name_df.head()
  ```
Kode tersebut memiliki luaran:

![TFIDF2](https://github.com/user-attachments/assets/fa34bb2d-7484-49b1-8078-65dbef0c04e0)


Penjelasan Penggunaan **TfidfVectorizer**

**`TfidfVectorizer(stop_words='english', max_features=100)`**:

- **`stop_words='english'`**: Mengabaikan kata-kata umum dalam bahasa Inggris (misalnya "the", "and", "is") yang tidak memberi informasi penting.
- **`max_features=100`**: Mengambil hanya 100 fitur (kata) teratas berdasarkan nilai TF-IDF tertinggi.

---

Fungsi Penting dalam Proses TF-IDF:

1. **`fit_transform()`**: Melakukan dua langkah:
   - **`fit()`**: Mempelajari vocabulari (kamus kata) yang ada di dalam kolom teks.
   - **`transform()`**: Mengubah teks ke dalam bentuk vektor berdasarkan fitur yang sudah dipelajari.

2. **`toarray()`**: Mengubah hasil yang awalnya berupa matriks sparse ke dalam array dua dimensi yang lebih mudah dibaca dan digunakan.

3. **`get_feature_names_out()`**: Mengambil nama-nama fitur (kata-kata) yang digunakan dalam vektor TF-IDF.

---

TF-IDF memberikan wawasan tentang seberapa penting genre tertentu dalam konteks anime tertentu. Genre dengan nilai yang lebih tinggi dianggap lebih relevan atau lebih penting dalam anime tersebut. Anime dengan genre yang dominan seperti Kimi no Na wa. (dengan genre supernatural) atau Fullmetal Alchemist: Brotherhood (dengan genre action dan adventure) menunjukkan bahwa nilai TF-IDF mencerminkan pentingnya genre-genre tersebut dalam anime tersebut, dibandingkan dengan anime lainnya dalam dataset.

### Cosine Similarity
- ```python
	print(cosine_sim_df.head())
  ```
Kode tersebut memiliki luaran:

![CosineSimilarity](https://github.com/user-attachments/assets/94260d4a-eabc-46a7-b0ed-450d1ef1cd13)

Cosine Similarity dalam konteks ini digunakan untuk membandingkan kemiripan antar anime berdasarkan genre mereka. Nilai mendekati 1.0 menunjukkan kemiripan yang tinggi, sedangkan nilai mendekati 0.0 menunjukkan perbedaan besar dalam genre. Anime seperti Kimi no Na wa. dan Fullmetal Alchemist: Brotherhood memiliki sedikit kemiripan karena genre yang sangat berbeda, sementara anime dalam satu franchise atau dengan tema serupa (seperti Gintama° dan Gintama') memiliki nilai kemiripan yang sangat tinggi. Steins;Gate sangat berbeda dari anime lain dalam dataset ini, yang tercermin dalam nilai 0.0 dengan anime lain yang memiliki genre berbeda seperti Kimi no Na wa.


## Evaluation
- ```python
	anime_name = 'Fullmetal Alchemist: Brotherhood'
	recommendations = get_recommendations(anime_name, cosine_sim_df, top_n=5)
	print(f"Rekomendasi untuk anime '{anime_name}':")
	print(recommendations)
  ```
Kode tersebut memiliki luaran:
  
![HasilRekomendasi](https://github.com/user-attachments/assets/ea64417a-a081-405b-9560-f513b376d5ee)

Rekomendasi ini menunjukkan bahwa anime yang memiliki genre yang mirip dengan Fullmetal Alchemist: Brotherhood (misalnya, dalam hal tema aksi, petualangan, drama, dan fantasi) akan mendapatkan skor Cosine Similarity yang lebih tinggi. Oleh karena itu, anime dengan skor yang lebih tinggi seperti Fullmetal Alchemist dan Fullmetal Alchemist: The Sacred Star of Milos lebih disarankan sebagai pilihan yang lebih relevan, sedangkan yang lain tetap dapat dianggap sebagai pilihan yang layak untuk penggemar genre serupa.

### Evaluasi Presisi
Perhitungan Presisi: Misalkan kita menetapkan threshold bahwa rekomendasi dengan Cosine Similarity > 0.8 dianggap relevan. Dari 5 rekomendasi, semuanya memiliki nilai Cosine Similarity di atas 0.8, yang berarti semuanya relevan. Dengan demikian, presisi untuk sistem rekomendasi ini adalah: Presisi = Jumlah rekomendasi relevan / Jumlah total rekomendasi = 5/5 = 1.0

Artinya, 100% dari rekomendasi yang diberikan relevan menurut threshold yang kita tentukan. Dengan demikian diharapkan dapat mencapai tujuan utama studi dilakukan yaitu Meningkatkan Pengalaman Pengguna.

## Referensi
[1] A. Sitanggang, "Sistem Rekomendasi Anime Menggunakan Metode Singular Value Decomposition (SVD) dan Cosine Similarity". Jurnal Teknologi Informasi. 2023 Nov 24;2(2):90-4.

[2] N.M. Roziqiin, M. Faisal "Sistem rekomendasi pemilihan anime menggunakan user-based collaborative filtering". JIPI (Jurnal Ilmiah Penelitian dan Pembelajaran Informatika). 2024 Feb 23;9(1):299-306.

[3] Putri HD, Faisal M. "Analyzing the effectiveness of collaborative filtering and content-based filtering methods in anime recommendation systems". Jurnal Komtika (Komputasi dan Informatika). 2023;7(2):124-33.
