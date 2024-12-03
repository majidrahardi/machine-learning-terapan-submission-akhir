# Laporan Proyek Akhir Machine Learning Terapan - Majid Rahardi

## Domain Proyek
Dalam beberapa tahun terakhir, anime telah berkembang menjadi bentuk hiburan global yang sangat populer, dengan jutaan penggemar di seluruh dunia. Seiring dengan meningkatnya jumlah judul anime yang tersedia di berbagai platform streaming seperti Crunchyroll, Netflix, dan Funimation, penggemar anime sering kali dihadapkan dengan pilihan yang sangat banyak. Hal ini membuat mereka kesulitan untuk menemukan anime yang sesuai dengan preferensi pribadi mereka. Di sinilah peran sistem rekomendasi menjadi sangat penting.

Sistem rekomendasi adalah alat yang menggunakan algoritma untuk membantu pengguna menemukan konten yang sesuai dengan minat atau preferensi mereka berdasarkan data yang ada. Dalam konteks anime, sistem rekomendasi bertujuan untuk menyarankan anime yang serupa dengan anime yang sudah ditonton atau yang dianggap disukai oleh pengguna, berdasarkan berbagai faktor seperti genre, tema, karakter, dan rating. Sistem ini berfungsi untuk meningkatkan pengalaman pengguna dengan menyederhanakan proses pencarian konten yang relevan dan memperkenalkan mereka pada judul anime baru yang mungkin belum mereka temui sebelumnya.

Salah satu teknik yang sering digunakan dalam sistem rekomendasi adalah Content-Based Filtering. Dalam pendekatan ini, algoritma menganalisis konten yang ada dalam database, seperti genre, deskripsi, atau karakteristik lain dari anime, dan membandingkannya dengan anime yang telah ditonton atau disukai oleh pengguna. Salah satu metode untuk mengukur kesamaan konten adalah TF-IDF (Term Frequency - Inverse Document Frequency), yang membantu mengukur seberapa penting suatu kata dalam dokumen (dalam hal ini, genre atau deskripsi anime). Selain itu, Cosine Similarity digunakan untuk menghitung seberapa mirip dua anime berdasarkan vektor TF-IDF mereka, yang memungkinkan sistem untuk memberikan rekomendasi yang relevan.

Secara keseluruhan, tujuan dari sistem rekomendasi anime adalah untuk memberikan pengalaman yang lebih personal dan efisien bagi pengguna dalam menemukan anime yang sesuai dengan selera mereka, yang pada akhirnya dapat meningkatkan kepuasan dan keterlibatan mereka dengan platform streaming anime.

[[7](https://jurnal.iaii.or.id/index.php/RESTI/article/view/5498)].

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
Tahap Data Preparation bertujuan untuk memastikan bahwa data yang digunakan dalam pembuatan model klasifikasi memiliki kualitas yang baik dan siap untuk diolah. Data yang bersih, bebas dari inkonsistensi, dan disiapkan dengan baik merupakan prasyarat penting untuk mendapatkan hasil analisis yang akurat dan model pembelajaran mesin yang optimal. Berikut adalah langkah-langkah yang dilakukan dalam proses persiapan data pada proyek klasifikasi jamur ini:

### Transformasi dan Encoding Data Kategorikal
Sebagian besar atribut dalam dataset adalah kategori, dengan nilai-nilai berbentuk simbol atau huruf. Agar data ini dapat digunakan dalam model pembelajaran mesin, diperlukan proses encoding untuk mengubah data kategorikal menjadi bentuk numerik. 

- ```python
	string_columns = [col_name for col_name, col_type in df.dtypes if col_type == 'string']
	indexers = [StringIndexer(inputCol=col_name, outputCol=col_name+"_index", handleInvalid="skip").fit(df) for col_name in string_columns]
	pipeline = Pipeline(stages=indexers)
	df_encoded = pipeline.fit(df).transform(df)
	df_encoded.select([col_name+"_index" for col_name in string_columns]).show()
  ```
  Kode tersebut memiliki luaran:
  
  | class_index | cap-shape_index | cap-surface_index | cap-color_index | bruises_index | odor_index | gill-attachment_index | gill-spacing_index | gill-size_index | gill-color_index | stalk-shape_index | stalk-root_index | stalk-surface-above-ring_index | stalk-surface-below-ring_index | stalk-color-above-ring_index | stalk-color-below-ring_index | veil-type_index | veil-color_index | ring-number_index | ring-type_index | spore-print-color_index | population_index | habitat_index |
   |-------------|-----------------|-------------------|-----------------|---------------|------------|-----------------------|--------------------|-----------------|------------------|-------------------|------------------|------------------------------|------------------------------|----------------------------|----------------------------|-----------------|------------------|-------------------|-----------------|-------------------------|------------------|---------------|
   | 1.0         | 0.0             | 1.0               | 0.0             | 1.0           | 6.0        | 0.0                   | 0.0                | 1.0             | 7.0              | 1.0               | 2.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 2.0                     | 2.0              | 4.0           |
   | 0.0         | 0.0             | 1.0               | 3.0             | 1.0           | 4.0        | 0.0                   | 0.0                | 0.0             | 7.0              | 1.0               | 3.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 1.0                     | 3.0              | 1.0           |
   | 0.0         | 3.0             | 1.0               | 4.0             | 1.0           | 5.0        | 0.0                   | 0.0                | 0.0             | 3.0              | 1.0               | 3.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 1.0                     | 3.0              | 5.0           |
   | 1.0         | 0.0             | 0.0               | 4.0             | 1.0           | 6.0        | 0.0                   | 0.0                | 1.0             | 3.0              | 1.0               | 2.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 2.0                     | 2.0              | 4.0           |
   | 0.0         | 0.0             | 1.0               | 1.0             | 0.0           | 0.0        | 0.0                   | 1.0                | 0.0             | 7.0              | 0.0               | 2.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 1.0             | 1.0                     | 4.0              | 1.0           |
   | 0.0         | 0.0             | 0.0               | 3.0             | 1.0           | 4.0        | 0.0                   | 0.0                | 0.0             | 3.0              | 1.0               | 3.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 2.0                     | 3.0              | 1.0           |
   | 0.0         | 3.0             | 1.0               | 4.0             | 1.0           | 4.0        | 0.0                   | 0.0                | 0.0             | 4.0              | 1.0               | 3.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 2.0                     | 3.0              | 5.0           |
   | 0.0         | 3.0             | 0.0               | 4.0             | 1.0           | 5.0        | 0.0                   | 0.0                | 0.0             | 3.0              | 1.0               | 3.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 1.0                     | 2.0              | 5.0           |
   | 1.0         | 0.0             | 0.0               | 4.0             | 1.0           | 6.0        | 0.0                   | 0.0                | 1.0             | 1.0              | 1.0               | 2.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 2.0                     | 0.0              | 1.0           |
   | 0.0         | 3.0             | 1.0               | 3.0             | 1.0           | 4.0        | 0.0                   | 0.0                | 0.0             | 4.0              | 1.0               | 3.0              | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0             | 2.0                     | 2.0              | 5.0           |

only showing top 20 rows

Output tersebut menunjukkan bahwa data awal telah dikodekan atau diubah menjadi indeks numerik, di mana setiap kolom atribut, seperti "class," "cap-shape," dan "odor," kini direpresentasikan dengan nilai numerik (misalnya, 0.0, 1.0, dll.) untuk memfasilitasi analisis dan pemrosesan lebih lanjut, seperti pelatihan model machine learning. Setiap nilai indeks tersebut mengacu pada kategori unik dari atribut yang sesuai, dan hanya 20 baris pertama dari data yang dikodekan yang ditampilkan sebagai sampel.

- ```python
	numeric_features = [t[0] for t in df.dtypes if t[1] in ['int','double']]
	numeric_summary = df.select(numeric_features).summary()
	numeric_summary.show(truncate=False)
  ```
  Kode tersebut memiliki luaran:

| summary | class_index | cap-shape_index | cap-surface_index | cap-color_index | bruises_index | odor_index | gill-attachment_index | gill-spacing_index | gill-size_index | gill-color_index | stalk-shape_index | stalk-root_index | stalk-surface-above-ring_index | stalk-surface-below-ring_index | stalk-color-above-ring_index | stalk-color-below-ring_index | veil-type_index | veil-color_index | ring-number_index | ring-type_index | spore-print-color_index | population_index | habitat_index |
|---------|-------------|-----------------|-------------------|-----------------|---------------|------------|-----------------------|--------------------|-----------------|------------------|-------------------|------------------|------------------------------|------------------------------|----------------------------|----------------------------|-----------------|------------------|-------------------|------------------|-------------------------|------------------|---------------|
| count  | 8124        | 8124            | 8124              | 8124            | 8124          | 8124       | 8124                  | 8124               | 8124            | 8124             | 8124              | 8124             | 8124                         | 8124                         | 8124                       | 8124                       | 8124            | 8124             | 8124              | 8124             | 8124                    | 8124             | 8124          |
| mean   | 0.482       | 0.777           | 0.887             | 1.785            | 0.416          | 1.453       | 0.026                 | 0.161              | 0.309           | 2.703            | 0.433             | 0.881             | 0.437                        | 0.536                        | 0.978                      | 1.018                      | 0.0             | 0.038            | 0.083             | 0.696            | 1.495                   | 1.064             | 1.356          |
| stddev | 0.500       | 0.876           | 0.821             | 1.650            | 0.493          | 1.916       | 0.159                 | 0.368              | 0.462           | 2.401            | 0.495             | 1.037             | 0.632                        | 0.779                        | 1.483                      | 1.523                      | 0.0             | 0.258            | 0.291             | 0.787            | 1.381                   | 1.392             | 1.544          |
| min    | 0.0         | 0.0             | 0.0               | 0.0              | 0.0            | 0.0         | 0.0                   | 0.0                | 0.0             | 0.0               | 0.0               | 0.0               | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0              | 0.0                     | 0.0              | 0.0           |
| 25%    | 0.0         | 0.0             | 0.0               | 0.0              | 0.0            | 0.0         | 0.0                   | 0.0                | 0.0             | 1.0               | 0.0               | 0.0               | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 0.0              | 0.0                     | 0.0              | 0.0           |
| 50%    | 0.0         | 1.0             | 1.0               | 1.0              | 0.0            | 1.0         | 0.0                   | 0.0                | 0.0             | 2.0               | 0.0               | 1.0               | 0.0                          | 0.0                          | 0.0                        | 0.0                        | 0.0             | 0.0              | 0.0               | 1.0              | 1.0                     | 1.0              | 1.0           |
| 75%    | 1.0         | 1.0             | 2.0               | 3.0              | 1.0            | 2.0         | 0.0                   | 0.0                | 1.0             | 4.0               | 1.0               | 1.0               | 1.0                          | 1.0                          | 1.0                        | 1.0                        | 0.0             | 0.0              | 0.0               | 1.0              | 2.0                     | 2.0              | 2.0           |
| max    | 1.0         | 5.0             | 3.0               | 9.0              | 1.0            | 8.0         | 1.0                   | 1.0                | 1.0             | 11.0              | 1.0               | 4.0               | 3.0                          | 3.0                          | 8.0                        | 8.0                        | 0.0             | 3.0              | 2.0               | 4.0              | 8.0                     | 5.0              | 6.0           |


Output tersebut memberikan ringkasan statistik dari kolom-kolom terkode dalam dataset, termasuk jumlah data (count), rata-rata (mean), standar deviasi (stddev), nilai minimum (min), kuartil pertama (25%), median (50%), kuartil ketiga (75%), dan nilai maksimum (max) untuk setiap kolom yang bertipe data numerik. Ini membantu memahami distribusi dan rentang nilai dari atribut terkode, yang mencakup informasi seperti "class_index" hingga "habitat_index," di mana nilai-nilai ini mewakili kategori yang telah diubah menjadi bentuk numerik untuk keperluan analisis dan model machine learning.

### Pemeriksaan Ketidakseimbangan Kelas
Mengingat bahwa target prediksi adalah label klasifikasi jamur (dapat dimakan atau beracun), penting untuk memeriksa keseimbangan distribusi kelas target. Ketidakseimbangan kelas yang signifikan dapat mempengaruhi performa model. Jika ditemukan, langkah-langkah penanganan seperti:
- **Synthetic Minority Oversampling Technique (SMOTE)**
- ```python
	class_counts_after = df_resampled.groupBy("class_index").count()
	print("Class counts after SMOTE:")
	class_counts_after.show()
  ```
  Kode tersebut memiliki luaran:

**Class counts after SMOTE:**
| class_index | count |
|-------------|-------|
| 0.0         | 4208  |
| 1.0         | 4208  |

Output tersebut menunjukkan hasil dari proses oversampling menggunakan SMOTE (Synthetic Minority Over-sampling Technique), di mana jumlah data untuk kedua kelas dalam kolom "class_index" kini seimbang, masing-masing memiliki 4208 catatan. Sebelumnya, kelas yang kurang terwakili telah ditingkatkan jumlahnya untuk mengatasi ketidakseimbangan kelas, sehingga dataset menjadi lebih seimbang dan dapat meningkatkan kinerja model machine learning dalam memprediksi kedua kelas secara adil.

### Correlation Analysis After Encoding and SMOTE
Analisis Korelasi adalah metode statistik yang digunakan untuk mengukur dan mengevaluasi hubungan antara dua variabel atau lebih. Tujuan utama dari analisis ini adalah untuk menentukan seberapa kuat hubungan tersebut dan apakah hubungan tersebut bersifat positif, negatif, atau netral.
- ```python
	#Menampilkan heatmap menggunakan Seaborn dan Matplotlib
	plt.figure(figsize=(12, 8))
	sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
	plt.title("Correlation Matrix Heatmap (Numeric Columns Only)")
	plt.show()
  ```
  Kode tersebut memiliki luaran:
  ![CorrelationAnalysis](https://github.com/user-attachments/assets/006b7361-5a54-4953-8541-e80b90179eef)

Gambar tersebut menunjukkan heatmap dari matriks korelasi yang merepresentasikan hubungan antara kolom-kolom numerik dalam dataset. Warna dalam heatmap berkisar dari -1 (biru tua) hingga +1 (merah tua), di mana angka yang lebih tinggi atau lebih dekat dengan +1 menunjukkan korelasi positif yang kuat, angka yang lebih rendah atau mendekati -1 menunjukkan korelasi negatif yang kuat, dan angka mendekati 0 menunjukkan korelasi yang lemah atau tidak ada korelasi. Diagonal utama semuanya merah tua dengan nilai +1 karena setiap variabel memiliki korelasi sempurna dengan dirinya sendiri. Ada beberapa korelasi signifikan yang dapat diamati, seperti hubungan kuat antara "ring-type_index" dan "ring-number_index," yang menunjukkan bahwa ada pola atau ketergantungan di antara atribut-atribut ini.


### Fitur Selection (Pearson Correlation)
- ```python
	# Menentukan ambang korelasi yang dianggap relevan
	threshold = 0.1
	# Memfilter fitur-fitur yang memiliki korelasi lebih besar dari ambang dengan class_index
	relevant_features = [col_name for col_name in correlation_df.index
                     if abs(correlation_df.loc[col_name, 'class_index']) > threshold
                     and col_name != "class_index"]

	# Menampilkan 5 fitur teratas yang dianggap relevan
	print("5 Fitur Teratas yang Dianggap Relevan terhadap class_index:")
	for feature in relevant_features[:5]:
   	 print(feature)
  	```
  Kode tersebut memiliki luaran:
  ```python
 	5 Fitur Teratas yang Dianggap Relevan terhadap class_index:
	cap-surface_index
	bruises_index
	odor_index
	gill-attachment_index
	gill-spacing_index
  ```
Output tersebut menunjukkan lima fitur teratas yang paling relevan terhadap "class_index," yaitu "cap-surface_index," "bruises_index," "odor_index," "gill-attachment_index," dan "gill-spacing_index." Ini berarti bahwa fitur-fitur ini memiliki hubungan yang lebih signifikan dengan variabel target "class_index" dibandingkan dengan fitur lainnya, yang dapat mempengaruhi prediksi atau klasifikasi dalam model machine learning. Relevansi ini dapat diukur melalui korelasi atau metode pemilihan fitur lainnya, dan fitur-fitur yang dipilih diharapkan memberikan kontribusi besar dalam meningkatkan kinerja model.

### Pemisahan Data
Data yang sudah bersih dan siap digunakan kemudian dibagi menjadi data latih (training set) dan data uji (testing set). Pembagian ini dilakukan untuk memastikan bahwa model dapat dievaluasi secara objektif, dengan mengukur kinerjanya pada data yang belum pernah dilihat sebelumnya.
- ```python
	# Membagi data menjadi train (50%), test (25%), dan validation (25%)
	train, temp = df_split.randomSplit([0.5, 0.5], seed=42)
	test, validation = temp.randomSplit([0.5, 0.5], seed=42)
	print("Training Dataset Count: " + str(train.count()))
	print("Test Dataset Count: " + str(test.count()))
	print("Validation Dataset Count: " + str(validation.count()))
  ```
  Kode tersebut memiliki luaran:
  ```python
 	Training Dataset Count: 4115
	Test Dataset Count: 2044
	Validation Dataset Count: 1965
  ```
Output tersebut menunjukkan bahwa dataset telah dibagi menjadi tiga subset: training, test, dan validation, menggunakan metode randomSplit dengan rasio 50:50 untuk pembagian awal, diikuti oleh pembagian 50:50 dari subset sementara untuk menghasilkan set test dan validation. Hasilnya, dataset training memiliki 4115 catatan, dataset test memiliki 2044 catatan, dan dataset validation memiliki 1965 catatan. Pembagian ini dilakukan untuk melatih model pada data training, menguji kinerjanya pada data test, dan memvalidasi atau menyetel model lebih lanjut menggunakan data validation.

## Modeling

- ```python
	# Inisialisasi model
	rf = RandomForestClassifier(featuresCol='features', labelCol='class_index')
	lr = LogisticRegression(featuresCol='features', labelCol='class_index')
	dt = DecisionTreeClassifier(featuresCol='features', labelCol='class_index')
	nb = NaiveBayes(featuresCol='features', labelCol='class_index')
	
	# Fit model pada data latih
	rf_model = rf.fit(train)
	lr_model = lr.fit(train)
	dt_model = dt.fit(train)
	nb_model = nb.fit(train)
	
	# Transformasi data uji dan validasi dengan model yang sudah dilatih
	rf_test_predictions = rf_model.transform(test)
	lr_test_predictions = lr_model.transform(test)
	dt_test_predictions = dt_model.transform(test)
	nb_test_predictions = nb_model.transform(test)
	
	rf_validation_predictions = rf_model.transform(validation)
	lr_validation_predictions = lr_model.transform(validation)
	dt_validation_predictions = dt_model.transform(validation)
	nb_validation_predictions = nb_model.transform(validation)
  ```
Kode tersebut menginisialisasi empat model klasifikasi dari pustaka PySpark ML, yaitu RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, dan NaiveBayes. 
Keempat model yang digunakan dalam proyek ini memiliki cara kerja yang berbeda. Semua model yang digunakan di sini menggunakan nilai parameter default yang telah ditentukan oleh pustaka yang digunakan yaitu pustaka PySpark ML.

### Random Forest:
Random Forest adalah algoritma ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan ketepatan prediksi dan mengurangi overfitting. Setiap pohon dalam hutan dibangun menggunakan subset acak dari data pelatihan dan subset acak dari fitur, sehingga meningkatkan variasi antar pohon dan membuat model lebih robust. Dengan parameter default seperti `n_estimators=100` dan `criterion='gini'`, Random Forest sangat efektif dalam menangani data besar dengan fitur yang banyak, serta memberikan hasil yang stabil meski sering kali lebih lambat dalam pelatihan dibandingkan model pohon tunggal.

#### Kelebihan Random Forest:
1. **Akurasi Tinggi** – Lebih akurat dibanding model tunggal.
2. **Tahan Overfitting** – Menggabungkan banyak pohon mengurangi overfitting.
3. **Resisten terhadap Noise** – Tidak mudah terpengaruh data yang salah atau acak.
4. **Menangani Banyak Fitur** – Efektif untuk dataset dengan fitur banyak.
5. **Estimasi Fitur Penting** – Bisa menilai pentingnya setiap fitur.

#### Kekurangan Random Forest:
1. **Kompleksitas Tinggi** – Butuh banyak komputasi dan memori.
2. **Interpretasi Sulit** – Hasilnya tidak mudah dipahami seperti pohon keputusan tunggal.
3. **Lambat untuk Prediksi Real-Time** – Kurang ideal untuk prediksi instan.
4. **Kurang Optimal untuk Data Waktu** – Tidak cocok untuk data berurutan atau time series.

### Logistic Regression:
Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memodelkan probabilitas suatu kelas berdasarkan kombinasi linier dari fitur yang ada. Meskipun bernama "regresi", model ini lebih sering digunakan untuk klasifikasi, dengan menggunakan fungsi sigmoid untuk mengubah log-odds menjadi probabilitas antara 0 dan 1. Dengan parameter default seperti `solver='lbfgs'` dan `max_iter=100`, Logistic Regression efisien dalam menangani masalah klasifikasi biner dan dapat diperbaiki dengan regularisasi untuk meningkatkan performa pada data yang kompleks.

#### Kelebihan Logistic Regression:
1. **Sederhana dan Mudah Diinterpretasi** – Modelnya sederhana dan mudah dipahami, sehingga memudahkan interpretasi hasil.
2. **Efisien untuk Dataset Kecil** – Logistic Regression bekerja baik dengan dataset yang lebih kecil dan memiliki fitur yang relevan.
3. **Cepat dan Ringan** – Proses training dan prediksi cepat, serta membutuhkan sumber daya komputasi yang lebih sedikit dibandingkan model kompleks.
4. **Probabilitas Output** – Menghasilkan nilai probabilitas, sehingga cocok untuk klasifikasi dengan pengukuran kepercayaan.

#### Kekurangan Logistic Regression:
1. **Tidak Efektif untuk Hubungan Non-Linear** – Logistic Regression hanya bisa menangani data yang memiliki hubungan linier; kurang efektif untuk hubungan non-linier.
2. **Rentan terhadap Outlier** – Outlier pada data dapat memengaruhi kinerja model, terutama tanpa normalisasi atau penanganan khusus.
3. **Tidak Ideal untuk Data yang Sangat Kompleks** – Kurang akurat pada data yang kompleks atau yang memiliki banyak fitur interaksi.
4. **Mengasumsikan Indepedensi Fitur** – Logistic Regression bekerja optimal jika fitur tidak saling bergantung, asumsi ini sering kali tidak realistis dalam data nyata.

### Decision Tree:
Decision Tree adalah algoritma klasifikasi yang membangun pohon keputusan dengan membagi data berdasarkan fitur-fitur yang paling informatif, diukur dengan kriteria seperti Gini Impurity atau Entropy. Proses ini berlanjut hingga data tidak dapat dibagi lebih lanjut atau mencapai kriteria berhenti yang ditentukan. Dengan menggunakan parameter default seperti `criterion='gini'`, `max_depth=None`, dan `min_samples_split=2`, Decision Tree mampu menangani data yang kompleks, meskipun cenderung mudah overfitting jika tidak diatur dengan baik.

#### Kelebihan Decision Tree:
1. **Mudah Diinterpretasi** – Struktur pohon membuat model ini mudah dipahami, visualisasi langsung menunjukkan jalur keputusan.
2. **Menangani Data Non-Linear** – Decision Tree dapat menangani data yang memiliki hubungan non-linier, cocok untuk berbagai jenis data.
3. **Tidak Perlu Banyak Pra-pemrosesan** – Model ini tidak memerlukan normalisasi atau penskalaan fitur.
4. **Bisa Menangani Fitur Kategorikal dan Numerik** – Decision Tree fleksibel dalam menangani tipe data yang berbeda dalam satu model.

#### Kekurangan Decision Tree:
1. **Rentan terhadap Overfitting** – Decision Tree cenderung mempelajari data secara detail hingga berlebihan, terutama tanpa pemangkasan (*pruning*).
2. **Sensitif terhadap Variasi Data** – Sedikit perubahan pada data dapat menyebabkan perubahan besar dalam struktur pohon.
3. **Kurang Optimal pada Dataset Besar** – Pohon yang sangat dalam bisa menjadi lambat dan memori intensif pada dataset besar.
4. **Cenderung Bias pada Fitur yang Dominan** – Decision Tree cenderung lebih terfokus pada fitur yang memiliki banyak kategori atau nilai tinggi.

### Naive Bayes:
Naive Bayes adalah model klasifikasi berbasis probabilistik yang mengasumsikan independensi antar fitur untuk menghitung probabilitas suatu kelas berdasarkan data yang ada. Menggunakan Teorema Bayes, model ini menghitung probabilitas kelas yang diberikan fitur dan memilih kelas dengan probabilitas tertinggi. Meskipun seringkali tidak realistis untuk mengasumsikan bahwa fitur-fitur saling independen, Naive Bayes sering kali memberikan hasil yang sangat baik, terutama pada masalah klasifikasi teks, dengan nilai parameter default seperti `var_smoothing=1e-9` untuk mencegah pembagian dengan nol.

#### Kelebihan Naive Bayes:
1. **Cepat dan Efisien** – Naive Bayes memiliki waktu training yang cepat, bahkan pada dataset yang besar.
2. **Sederhana dan Mudah Diimplementasikan** – Model ini sangat mudah diimplementasikan dan interpretasinya sederhana.
3. **Performa Baik pada Data Kecil** – Naive Bayes bekerja baik pada dataset yang kecil dan terstruktur dengan baik.
4. **Bekerja Baik pada Data Kategorikal** – Sangat cocok untuk data kategorikal, seperti klasifikasi teks (misalnya, klasifikasi email sebagai spam atau bukan).

#### Kekurangan Naive Bayes:
1. **Mengasumsikan Indepedensi Fitur** – Asumsi bahwa setiap fitur bersifat independen tidak selalu realistis, terutama pada data nyata.
2. **Sensitif terhadap Data Tidak Relevan** – Naive Bayes bisa terpengaruh oleh fitur yang kurang relevan, sehingga perlu pemilihan fitur yang cermat.
3. **Rentan terhadap Data Tak Terlihat** – Kinerja bisa turun jika terdapat data atau kombinasi kategori yang belum pernah terlihat selama pelatihan.
4. **Tidak Cocok untuk Data dengan Interaksi Fitur yang Kuat** – Kurang efektif jika fitur memiliki korelasi tinggi, karena asumsi independensi tidak terpenuhi.

## Evaluation
### Confusion Matrix

Confusion Matrix adalah tabel yang digunakan untuk mengevaluasi kinerja model klasifikasi dengan menghitung jumlah prediksi yang benar dan salah dari setiap kelas. Tabel ini menunjukkan bagaimana model memprediksi setiap kelas dan membantu mengidentifikasi di mana kesalahan terjadi.

### Komponen Confusion Matrix

Confusion Matrix terdiri dari empat komponen utama:

1. **True Positive (TP)**: Prediksi yang benar di mana model memprediksi positif dan hasil sebenarnya juga positif.
2. **True Negative (TN)**: Prediksi yang benar di mana model memprediksi negatif dan hasil sebenarnya juga negatif.
3. **False Positive (FP)**: Prediksi salah di mana model memprediksi positif, tetapi hasil sebenarnya adalah negatif (*Type I Error*).
4. **False Negative (FN)**: Prediksi salah di mana model memprediksi negatif, tetapi hasil sebenarnya adalah positif (*Type II Error*).

### Struktur Confusion Matrix:

|                 | Prediksi Positif | Prediksi Negatif |
|-----------------|------------------|------------------|
| **Aktual Positif** | True Positive (TP)  | False Negative (FN) |
| **Aktual Negatif** | False Positive (FP) | True Negative (TN)  |

### Metrik dari Confusion Matrix

Confusion Matrix memungkinkan kita menghitung berbagai metrik evaluasi penting, antara lain:

- **Akurasi**: Proporsi prediksi yang benar dari seluruh prediksi. Akurasi = TP + TN / TP + TN + FP + FN

- **Presisi**: Proporsi prediksi positif yang benar dari seluruh prediksi positif. Presisi = TP / TP + FP

- **Recall (Sensitivitas)**: Proporsi kasus positif yang teridentifikasi dengan benar oleh model. Recall = TP / TP + FN
  
- **F1-Score**: Rata-rata harmonis dari presisi dan recall, mengukur keseimbangan antara keduanya. F1-Score = 2 (Presisi x Recall) / Presisi + Recall

### Hasil Confusion Matrix

- Confusion Matrix Random Forest
  
![Confusion Matrix Random Forest](https://github.com/user-attachments/assets/4887a649-59b6-4080-81e1-0229f1665152)


- Confusion Matrix Logistic Regression
  
![Confusion Matrix Logistic Regression](https://github.com/user-attachments/assets/a524b311-af1b-4a49-8407-d62f2894012d)

  
- Confusion Matrix Decision Tree
  
![Confusion Matrix Decision Tree](https://github.com/user-attachments/assets/f50d58bb-19cb-487d-b326-ce415c09e928)

  
- Confusion Matrix Naive Bayes
  
![Confusion Matrix Naive Bayes](https://github.com/user-attachments/assets/038cda72-d1cd-485b-8427-1fdf662f193c)


### Kesimpulan
- ```python
	print_metrics('Random Forest', rf_val_metrics, 'Validation')
	print_metrics('Logistic Regression', lr_val_metrics, 'Validation')
	print_metrics('Decision Tree', dt_val_metrics, 'Validation')
	print_metrics('Naive Bayes', nb_val_metrics, 'Validation')
	print_metrics('Random Forest', rf_test_metrics, 'Test')
	print_metrics('Logistic Regression', lr_test_metrics, 'Test')
	print_metrics('Decision Tree', dt_test_metrics, 'Test')
	print_metrics('Naive Bayes', nb_test_metrics, 'Test')
  ```
  Kode tersebut memiliki luaran:
    ```python
  	Random Forest (Test)
	Accuracy: 98.87%
	Precision: 98.90%
	Recall: 98.87%
	F1-Score: 98.87%
	
	Logistic Regression (Test)
	Accuracy: 84.49%
	Precision: 84.92%
	Recall: 84.49%
	F1-Score: 84.47%
	
	Decision Tree (Test)
	Accuracy: 98.24%
	Precision: 98.24%
	Recall: 98.24%
	F1-Score: 98.24%
	
	Naive Bayes (Test)
	Accuracy: 75.54%
	Precision: 75.59%
	Recall: 75.54%
	F1-Score: 75.49%
  ```

Berdasarkan hasil evaluasi model yang diterapkan pada dataset ini, Random Forest menunjukkan kinerja terbaik dengan akurasi, presisi, recall, dan F1-Score masing-masing mencapai 98.87%. Angka ini menunjukkan bahwa model Random Forest dapat dengan sangat baik mengklasifikasikan data, dengan kesalahan klasifikasi yang sangat minimal.

Model Decision Tree juga menunjukkan kinerja yang sangat baik, dengan hasil yang hampir serupa dengan Random Forest, yaitu akurasi, presisi, recall, dan F1-Score masing-masing mencapai 98.24%. Meskipun sedikit lebih rendah dibandingkan Random Forest, Decision Tree tetap memberikan performa yang sangat baik dalam klasifikasi data.

Di sisi lain, Logistic Regression memiliki hasil yang cukup baik dengan akurasi 84.49%, presisi 84.92%, recall 84.49%, dan F1-Score 84.47%. Meskipun lebih rendah dibandingkan Random Forest dan Decision Tree, Logistic Regression masih menunjukkan performa yang solid, namun kurang optimal dibandingkan dengan model lainnya.

Terakhir, Naive Bayes menunjukkan hasil yang lebih rendah dengan akurasi 75.54%, presisi 75.59%, recall 75.54%, dan F1-Score 75.49%. Hasil ini mengindikasikan bahwa model Naive Bayes kurang efektif dibandingkan dengan model lainnya dalam hal klasifikasi data pada dataset ini.

Secara keseluruhan, Random Forest dan Decision Tree adalah model yang lebih unggul dalam mengklasifikasikan data, sedangkan Logistic Regression dan Naive Bayes memiliki performa yang lebih rendah. Oleh karena itu, untuk aplikasi yang membutuhkan akurasi dan performa tinggi, Random Forest atau Decision Tree akan menjadi pilihan yang lebih baik.

Dalam laporan ini, kita mengevaluasi dampak dari model klasifikasi jamur yang dikembangkan terhadap Business Understanding yang telah diidentifikasi. Tujuan dari model ini adalah untuk mengatasi tantangan kritis dalam mengklasifikasikan jamur sebagai "dapat dimakan" atau "beracun," mengingat bahwa kesalahan dalam identifikasi dapat berdampak fatal bagi kesehatan manusia. Model ini dirancang untuk menjawab problem statement yang sangat spesifik: memberikan sistem klasifikasi yang akurat dan efisien berdasarkan karakteristik fisik jamur, yang sering menjadi dasar dalam identifikasi manual yang memakan waktu dan memerlukan keahlian tinggi.

Evaluasi menunjukkan bahwa model yang dikembangkan, khususnya algoritma seperti Random Forest dan Decision Tree, berhasil mencapai tingkat akurasi yang tinggi (lebih dari 98%) dalam mengklasifikasikan jamur. Ini menunjukkan bahwa model sudah cukup menjawab problem statement dengan memberikan solusi yang meminimalkan risiko kesalahan identifikasi. Dengan tingkat akurasi yang tinggi, model ini juga berhasil mencapai tujuan untuk meningkatkan keamanan publik. Pengguna umum, seperti kolektor jamur, dapat lebih percaya diri dalam membedakan jamur beracun dari yang dapat dimakan, yang berpotensi mengurangi insiden keracunan.

Dampak dari solution statement yang direncanakan juga cukup signifikan. Melalui pemrosesan data yang teliti dan pengembangan model yang dioptimalkan, model ini mampu memberikan prediksi yang akurat dengan memanfaatkan atribut fisik yang sudah tersedia dalam dataset. Implementasi alat identifikasi berbasis model ini akan sangat berguna, karena tidak hanya mengotomatisasi proses klasifikasi yang kompleks tetapi juga memberikan kemudahan akses bagi masyarakat umum tanpa memerlukan keahlian khusus. Dengan demikian, solusi ini memiliki dampak positif yang nyata dalam meningkatkan keselamatan dan efisiensi identifikasi jamur.

## Referensi
[1] O. Tarawneh, M. Tarawneh, Y. Sharrab, and M. Husni, "Mushroom classification using machine-learning techniques," in AIP Conference Proceedings, vol. 2979, no. 1, Oct. 2023.

[2] Y. Wang, J. Du, H. Zhang, and X. Yang, "Mushroom toxicity recognition based on multigrained cascade forest," Scientific Programming, vol. 2020, no. 1, pp. 1-10, 2020.

[3] Kaggle Mushroom Classification Dataset Documentation, [Online]. Available: https://www.kaggle.com/uciml/mushroom-classification. Accessed: Nov. 3, 2024.

[4] H. Ujir, I. Hipiny, M. H. Bolhassan, K. N. Fazira Ku Azir, and S. A. Ali, "Automating Mushroom Culture Classification: A Machine Learning Approach," International Journal of Advanced Computer Science & Applications, vol. 15, no. 4, 2024.

[5] X. Guo, "Research on Mushroom Image Classification Algorithm Based on Deep Sparse Dictionary Learning," Academic Journal of Science and Technology, vol. 9, no. 1, pp. 235-240, 2024.

[6] R. Sahu, S. Pandey, R. Verma, and P. Pandey, "Ensemble Learning based Classification of Edible and Poisonous Agaricus Mushrooms," in 2024 Fourth International Conference on Advances in Electrical, Computing, Communication and Sustainable Technologies (ICAECT), Jan. 2024, pp. 1-7.

[7] L. Farokhah and S. Y. Riska, "Analysis and Development of Eight Deep Learning Architectures for the Classification of Mushrooms," Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), vol. 8, no. 1, pp. 142-149, 2024.

