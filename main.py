

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


#### Veri Setinin yüklenmesi

df = pd.read_csv("Employee.csv")


pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.width', 1000)

df.head()

df.columns

### Veri Seti Kolonlarını Türkçe İsimleri
- Age: Yaş
- Attrition: İşten Ayrılma
- BusinessTravel: İş Seyahati
- DailyRate: Günlük Ücret
- Department: Departman
- DistanceFromHome: Evden Uzaklık
- Education: Eğitim
- EducationField: Eğitim Alanı
- EmployeeCount: Çalışan Sayısı
- EmployeeNumber: Çalışan Numarası
- EnvironmentSatisfaction: Çevre Memnuniyeti
- Gender: Cinsiyet
- HourlyRate: Saatlik Ücret
- JobInvolvement: İş Katılımı
- JobLevel: İş Seviyesi
- JobRole: İş Rolü
- JobSatisfaction: İş Memnuniyeti
- MaritalStatus: Medeni Durum
- MonthlyIncome: Aylık Gelir
- MonthlyRate: Aylık Ücret
- NumCompaniesWorked: Çalışılan Şirket Sayısı
- Over18: 18 Yaş Üzeri
- OverTime: Fazla Mesai
- PercentSalaryHike: Maaş Artış Yüzdesi
- PerformanceRating: Performans Değerlendirmesi
- RelationshipSatisfaction: İlişki Memnuniyeti
- StandardHours: Standart Saatler
- StockOptionLevel: Hisse Senedi Opsiyon Seviyesi
- TotalWorkingYears: Toplam Çalışma Yılları
- TrainingTimesLastYear: Geçen Yılki Eğitim Süreleri
- WorkLifeBalance: İş-Yaşam Dengesi
- YearsAtCompany: Şirkette Geçen Yıllar
- YearsInCurrentRole: Mevcut Roldeki Yıllar
- YearsSinceLastPromotion: Son Terfiden Bu Yana Geçen Yıllar
- YearsWithCurrManager: Mevcut Yöneticideki Yıllar

# Orijinal kolon adlarını Türkçeye çeviren bir sözlük oluşturun
column_mapping = {
    'Age': 'Yaş',
    'Attrition': 'İşten_Ayrılma',
    'BusinessTravel': 'İş_Seyahati',
    'DailyRate': 'Günlük_Ücret',
    'Department': 'Departman',
    'DistanceFromHome': 'Evden_Uzaklık',
    'Education': 'Eğitim',
    'EducationField': 'Eğitim_Alanı',
    'EmployeeCount': 'Çalışan_Sayısı',
    'EmployeeNumber': 'Çalışan_Numarası',
    'EnvironmentSatisfaction': 'Çevre_Memnuniyeti',
    'Gender': 'Cinsiyet',
    'HourlyRate': 'Saatlik_Ücret',
    'JobInvolvement': 'İş_Katılımı',
    'JobLevel': 'İş_Seviyesi',
    'JobRole': 'İş_Rolü',
    'JobSatisfaction': 'İş_Memnuniyeti',
    'MaritalStatus': 'Medeni_Durum',
    'MonthlyIncome': 'Aylık_Gelir',
    'MonthlyRate': 'Aylık_Ücret',
    'NumCompaniesWorked': 'Çalışılan_Şirket_Sayısı',
    'Over18': '18_Yaş_Üzeri',
    'OverTime': 'Fazla_Mesai',
    'PercentSalaryHike': 'Maaş_Artış_Yüzdesi',
    'PerformanceRating': 'Performans_Değerlendirmesi',
    'RelationshipSatisfaction': 'İlişki_Memnuniyeti',
    'StandardHours': 'Standart_Saatler',
    'StockOptionLevel': 'Hisse_Senedi_Opsiyon_Seviyesi',
    'TotalWorkingYears': 'Toplam_Çalışma_Yılları',
    'TrainingTimesLastYear': 'Geçen_Yılki_Eğitim_Süreleri',
    'WorkLifeBalance': 'İş_Yaşam_Dengesi',
    'YearsAtCompany': 'Şirkette_Geçen_Yıllar',
    'YearsInCurrentRole': 'Mevcut_Roldeki_Yıllar',
    'YearsSinceLastPromotion': 'Son_Terfiden_Bu_Yana_Geçen_Yıllar',
    'YearsWithCurrManager': 'Mevcut_Yöneticideki_Yıllar'
}

# Kolon adlarını değiştirin
df.rename(columns=column_mapping, inplace=True)

# Değiştirilen kolon adlarını kontrol edin
print(df.columns)


df.head()

#### Veri Seti Özelliklerinin İncelenmesi

print(f"Veri Seti {df.shape[0]} satır ve {df.shape[1]} sütundan oluşmaktadır. {df.ndim} boyutludur ve toplamda {df.size} veri (hücre) bulunmaktadır.")

df.info()

# Object tipindeki öznitelikleri ve sayısını göster
object_cols = df.select_dtypes(include='object').columns
print(f"Object Tipindeki Öznitelikler: {list(object_cols)}")
print(f"Sayısı: {len(object_cols)}\n")

# Int veya Float tipindeki öznitelikleri ve sayısını göster
numeric_cols = df.select_dtypes(exclude='object').columns
print(f"Int veya Float Tipindeki Öznitelikler: {list(numeric_cols)}")
print(f"Sayısı: {len(numeric_cols)}")

#### Veri Setinin İstatistik ÖZelliklerini İnceleme

df.describe().T

#### Veri seti genel olarak iyi bir dağılıma sahip ve eksik veri bulunmuyor.

### Eksik verilerin tespiti

df.isna().sum().sum()

# FutureWarning uyarılarını devre dışı bırakın
warnings.simplefilter(action='ignore', category=FutureWarning)

# Sonsuz değerleri NaN olarak değiştirin
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

# Eksik veri sayısını kontrol edin
print("Eksik veri sayısı:\n", df.isna().sum())

# Sayısal değişkenleri belirleyin
numerical_vars = [
    'Yaş', 'Günlük_Ücret', 'Evden_Uzaklık', 'Eğitim', 'Çevre_Memnuniyeti', 'Saatlik_Ücret',
    'İş_Katılımı', 'İş_Seviyesi', 'İş_Memnuniyeti', 'Aylık_Gelir', 'Aylık_Ücret',
    'Çalışılan_Şirket_Sayısı', 'Maaş_Artış_Yüzdesi', 'Performans_Değerlendirmesi',
    'İlişki_Memnuniyeti', 'Hisse_Senedi_Opsiyon_Seviyesi', 'Toplam_Çalışma_Yılları',
    'Geçen_Yılki_Eğitim_Süreleri', 'İş_Yaşam_Dengesi', 'Şirkette_Geçen_Yıllar',
    'Mevcut_Roldeki_Yıllar', 'Son_Terfiden_Bu_Yana_Geçen_Yıllar', 'Mevcut_Yöneticideki_Yıllar'
]


Veri setinde eksik veri bulunmamaktadır.

### Veri Sütunlarındaki Benzersiz Değerlerin Sayısı

# Her sütundaki benzersiz değer sayısını hesapla
unique_values = df.nunique()

# Benzersiz değer sayısını DataFrame olarak göster
unique_values_df = unique_values.reset_index()
unique_values_df.columns = ['Column', 'UniqueValues']

# Sonuçları yazdır
print(unique_values_df)


# "Çalışan Numarası" sütunundaki benzersiz değer sayısını hesaplayalım
unique_employee_numbers = df['Çalışan_Numarası'].nunique()
print("Benzersiz Çalışan Sayısı:", unique_employee_numbers)

### Analiz İçin Gereksiz Sütunları Çıkarma

# Sabit değerlere sahip sütunları tespit et
constant_columns = [col for col in df.columns if df[col].nunique() == 1]

# Sabit değerlere sahip sütunları çıkar
df_cleaned = df.drop(columns=constant_columns)

print("Çıkarılan sütunlar:", constant_columns)
print("Temizlenmiş veri seti sütunları:", df_cleaned.columns)


# Veri setindeki her bir sütunun değerlerinin frekansını inceleyelim
frequencies = {}

# Her sütundaki değerlerin frekanslarını hesaplayalım
for column in df_cleaned.columns:
    frequencies[column] = df_cleaned[column].value_counts()

# Her sütunun ilk 10 frekansını 5 grafik olacak şekilde bir arada görselleştirelim
columns = list(frequencies.keys())
num_columns = 5
num_rows = len(columns) // num_columns + (1 if len(columns) % num_columns > 0 else 0)

fig, axes = plt.subplots(num_rows, num_columns, figsize=(25, num_rows * 5))
axes = axes.flatten()

for i, column in enumerate(columns):
    freq = frequencies[column].head(10)
    axes[i].bar(freq.index, freq.values)
    axes[i].set_title(f'Frekanslar için {column} sütunu')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frekans')

# Kalan boş grafikleri gizle
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


  Frekans Analizine Göre;
- 35 yaşındaki çalışanlar en yaygın yaş grubudur.
- Çoğu çalışan işten ayrılmamış durumda (No), ancak belirli bir kısmı işten ayrılmıştır (Yes).
- Çalışanların çoğu nadiren iş seyahati yapmaktadır (Travel_Rarely)
- 691 günlük ücreti olan çalışanlar daha yaygındır.
- Araştırma ve Geliştirme (Research & Development) departmanı en fazla çalışanı içerirken, Satış (Sales) ve İnsan Kaynakları (Human Resources) departmanları daha az sayıda çalışana sahiptir.
- Çoğu çalışan 2 km uzaklıkta yaşarken, 1 km ve 3 km gibi diğer mesafeler de yaygındır.
- Eğitim seviyesi 3 en yaygın olanıdır.
- Yaşam Bilimleri (Life Sciences) en yaygın eğitim alanıdır.
- Çalışanların çevre memnuniyet seviyesi 3 en yaygın olanıdır.
- Erkek çalışanlar (Male) kadın çalışanlardan (Female) daha yaygındır.
- 66 saatlik ücreti olan çalışanlar daha yaygındır.
- İş katılımı seviyesi 3 en yaygın olanıdır.
- Çalışanların iş seviyesi 1 en yaygın olanıdır.
- Satış Yöneticisi (Sales Executive) en yaygın iş rolüdür.
- Çalışanların iş memnuniyet seviyesi 4 en yaygın olanıdır.
- Evli çalışanlar (Married) daha yaygındır.
- Belirli gelir aralıkları diğerlerine göre daha yaygındır.
- 2342 aylık ücreti olan çalışanlar daha yaygındır.
- Çoğu çalışan sadece bir şirkette çalışmıştır.
- Çoğu çalışan fazla mesai yapmamaktadır.
- Maaş artış yüzdeleri arasında belirli artış yüzdeleri daha yaygın.
- Performans seviyesi 3 en yaygın olanıdır.
- İlişki Memnuniyet seviyesi 3 en yaygın olanıdır.
- Hisse senedi seviyesi 0 en yaygın olanıdır.
- 10 yıl çalışma süresi daha yaygındır.
- Eğitim süresi 3 en yaygın olanıdır.
- İş yaşam dengesi seviyesi 3 en yaygındır.
- 5 yıl şirkette çalışma süresi daha yaygındır.
- 2 yıl mevcut rolde çalışma süresi daha yaygındır.
- 0 yıl terfi süresi daha yaygındır.
- 2 yıl mevcut yöneticide çalışma süresi daha yaygındır.

df['Cinsiyet'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Cinsiyete Göre Çalışan Dağılımı')
plt.show()


sns.pairplot(df[['Yaş', 'Aylık_Gelir', 'İş_Memnuniyeti']])
plt.suptitle('Çift Değişkenli Dağılım Grafikleri')
plt.show()

- Yaş: Yaşın iş memnuniyeti veya aylık gelir üzerinde belirgin bir etkisi yok gibi görünüyor. Çalışanların yaşı ne olursa olsun, aylık gelir ve iş memnuniyeti değişkenlik gösteriyor.
- Aylık Gelir: Aylık gelir ile iş memnuniyeti arasında belirgin bir korelasyon bulunmamakta. Yüksek gelirli çalışanlar da düşük gelirli çalışanlar da çeşitli memnuniyet seviyelerinde yer almakta.
- İş Memnuniyeti: İş memnuniyeti seviyeleri oldukça çeşitli ve dağılımlar geniş. Bu, çalışan memnuniyetinin tek bir faktöre bağlı olmadığını, birçok değişkenin bu durumu etkileyebileceğini göstermekte.

### İşten Ayrılma Oranı Analizi

- Genel İşten Ayrılma Oranı Analizi

# Genel işten ayrılma oranı analizi
genel_terk_orani = df_cleaned['İşten_Ayrılma'].value_counts(normalize=True) * 100
print("Genel İşten Ayrılma Oranı:")
print(genel_terk_orani)


- Departmana Göre İşten Ayrılma Oranı Analizi

# Departmana göre işten ayrılma oranı analizi
departmana_gore_terk_orani = df_cleaned.groupby('Departman')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nDepartmana Göre İşten Ayrılma Oranı:")
print(departmana_gore_terk_orani)

- İş Rolüne Göre İşten Ayrılma Oranı Analizi

# İş rolüne göre işten ayrılma oranı analizi
is_rolune_gore_terk_orani = df_cleaned.groupby('İş_Rolü')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nİş Rolüne Göre İşten Ayrılma Oranı:")
print(is_rolune_gore_terk_orani)

- İş Seyahati Sıklığına Göre İşten Ayrılma Oranı Analizi

# İş seyahati sıklığına göre işten ayrılma oranı analizi
is_seyahati_gore_terk_orani = df_cleaned.groupby('İş_Seyahati')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nİş Seyahati Sıklığına Göre İşten Ayrılma Oranı:")
print(is_seyahati_gore_terk_orani)

### Memnuniyet ve Performans Analizi

- İş Memnuniyeti ve İşten Ayrılma

# İş memnuniyeti ve işten ayrılma analizi
memnuniyet_terk_orani = df_cleaned.groupby('İş_Memnuniyeti')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nİş Memnuniyeti ve İşten Ayrılma Oranı:")
print(memnuniyet_terk_orani)

- Çevre Memnuniyeti ve İşten Ayrılma

# Çevre memnuniyeti ve işten ayrılma analizi
cevre_memnuniyeti_terk_orani = df_cleaned.groupby('Çevre_Memnuniyeti')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nÇevre Memnuniyeti ve İşten Ayrılma Oranı:")
print(cevre_memnuniyeti_terk_orani)


- İlişki Memnuniyeti ve İşten Ayrılma

# İlişki memnuniyeti ve işten ayrılma analizi
iliskı_memnuniyeti_terk_orani = df_cleaned.groupby('İlişki_Memnuniyeti')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nİlişki Memnuniyeti ve İşten Ayrılma Oranı:")
print(iliskı_memnuniyeti_terk_orani)

- Performans Değerlendirmesi ve İşten Ayrılma

# Performans değerlendirmesi ve işten ayrılma analizi
performans_terk_orani = df_cleaned.groupby('Performans_Değerlendirmesi')['İşten_Ayrılma'].value_counts(normalize=True).unstack() * 100
print("\nPerformans Değerlendirmesi ve İşten Ayrılma Oranı:")
print(performans_terk_orani)

### Korelasyon Analizi

- İşten Ayrılma ile Diğer Değişkenler Arasındaki Korelasyon

# Veri tiplerini kontrol edelim
print(df_cleaned.dtypes)


df_cleaned['İşten_Ayrılma'] = df_cleaned['İşten_Ayrılma'].apply(lambda x: 1 if x == 'Yes' else 0)
df_cleaned['İş_Seyahati'] = df_cleaned['İş_Seyahati'].map({'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2})
df_cleaned['Departman'] = df_cleaned['Departman'].map({'Research & Development': 0, 'Sales': 1, 'Human Resources': 2})
df_cleaned['Eğitim_Alanı'] = df_cleaned['Eğitim_Alanı'].astype('category').cat.codes
df_cleaned['Cinsiyet'] = df_cleaned['Cinsiyet'].map({'Male': 0, 'Female': 1})
df_cleaned['İş_Rolü'] = df_cleaned['İş_Rolü'].astype('category').cat.codes
df_cleaned['Medeni_Durum'] = df_cleaned['Medeni_Durum'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
df_cleaned['Fazla_Mesai'] = df_cleaned['Fazla_Mesai'].map({'Yes': 1, 'No': 0})



df_cleaned.dtypes

# Sayısal sütunları seçelim ve yalnızca sayısal sütunları içeren yeni bir veri çerçevesi oluşturalım
numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64', 'int'])

# Korelasyon matrisini hesaplayalım
correlation_matrix = numeric_df.corr()

# Korelasyon matrisini görselleştirelim
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()


- Pozitif Korelasyonlar:
Yaş ve Toplam Çalışma Yılları (0.68)
Yaş arttıkça, toplam çalışma yılları da artmaktadır.
Yaş ve Şirkette Geçen Yıllar (0.31)
Yaşlı çalışanlar şirkette daha uzun süre kalma eğilimindedir.
Toplam Çalışma Yılları ve Şirkette Geçen Yıllar (0.53)
Deneyimli çalışanlar genellikle daha uzun süre aynı şirkette çalışır.
İş Memnuniyeti ve Çevre Memnuniyeti (0.31)
Çevre memnuniyeti arttıkça iş memnuniyeti de artar.
Mevcut Roldeki Yıllar ve Şirkette Geçen Yıllar (0.77)
Şirkette daha uzun süre kalan çalışanlar genellikle aynı rolde de uzun süre kalır.
Son Terfiden Bu Yana Geçen Yıllar ve Şirkette Geçen Yıllar (0.55)
Şirkette daha uzun süre kalan çalışanların son terfilerinden bu yana daha fazla yıl geçmiştir.
Mevcut Yöneticideki Yıllar ve Mevcut Roldeki Yıllar (0.51)
Çalışanlar aynı yöneticide uzun süre kaldıklarında aynı rolde de uzun süre kalırlar.
- Dikkat Çeken İlişkiler:
Maaş artış yüzdesi ve performans değerlendirmesi arasında pozitif bir ilişki vardır (0.15).
Bu, yüksek performans gösteren çalışanların daha yüksek maaş artışları aldığını gösterir.
İş memnuniyeti ile çevre memnuniyeti arasında pozitif bir ilişki bulunmaktadır (0.31).
Bu, çevresinden memnun olan çalışanların işlerinden de memnun olduklarını gösterir.
- İşten Ayrılma ile İlişkiler:
İşten Ayrılma ile Maaş Artış Yüzdesi (-0.16):
Maaş artış yüzdesi arttıkça işten ayrılma oranı azalır.
İşten Ayrılma ile Aylık Gelir (-0.16):
Aylık gelir arttıkça işten ayrılma oranı azalır.
Sonuç:
Bu korelasyon matrisi, yaş, çalışma yılları ve memnuniyetin işten ayrılma üzerinde önemli etkileri olduğunu göstermektedir. Yüksek maaş artışları ve aylık gelirler işten ayrılma oranlarını düşürebilirken, çevre ve iş memnuniyeti çalışan bağlılığını artırabilir. Bu bulgular, çalışan memnuniyetini artırmak ve işten ayrılma oranlarını azaltmak için stratejik kararlar almak için kullanılabilir.









#### Performans ve İşten Ayrılma Arasındaki Korelasyon

# Performans değerlendirmesi ile işten ayrılma arasındaki korelasyon
performans_terk_korelasyonu = correlation_matrix.loc['Performans_Değerlendirmesi', 'İşten_Ayrılma']
print(f"Performans Değerlendirmesi ve İşten Ayrılma Arasındaki Korelasyon: {performans_terk_korelasyonu}")

# Performans değerlendirmesi ve işten ayrılma analizi
sns.boxplot(x='İşten_Ayrılma', y='Performans_Değerlendirmesi', data=df_cleaned)
plt.title('Performans Değerlendirmesi ve İşten Ayrılma Arasındaki İlişki')
plt.xlabel('İşten Ayrılma')
plt.ylabel('Performans Değerlendirmesi')
plt.show()

#### İşten Ayrılma Nedenleri Analizi

- İşten Ayrılma ile İlişkili Faktörlerin Korelasyonunu İnceleme

# İşten ayrılma ile diğer değişkenler arasındaki korelasyonu yazdıralım
attrition_corr = correlation_matrix['İşten_Ayrılma'].sort_values(ascending=False)
print(attrition_corr)

- Fazla mesai ve evden uzaklık, işten ayrılma olasılığını artıran faktörlerdir.
- Negatif korelasyon değerleri, değişken arttıkça işten ayrılma olasılığının azaldığını gösterir.
- Performans değerlendirmesi ile işten ayrılma arasında çok zayıf bir pozitif ilişki vardır, bu da performans değerlendirmesinin işten ayrılma üzerinde neredeyse hiç etkisi olmadığını gösterir.

# Sayısal değişkenleri belirleyin
numerical_vars = [
    'Yaş', 'Günlük_Ücret', 'Evden_Uzaklık', 'Eğitim', 'Çevre_Memnuniyeti', 'Saatlik_Ücret',
    'İş_Katılımı', 'İş_Seviyesi', 'İş_Memnuniyeti', 'Aylık_Gelir', 'Aylık_Ücret',
    'Çalışılan_Şirket_Sayısı', 'Maaş_Artış_Yüzdesi', 'Performans_Değerlendirmesi',
    'İlişki_Memnuniyeti', 'Hisse_Senedi_Opsiyon_Seviyesi', 'Toplam_Çalışma_Yılları',
    'Geçen_Yılki_Eğitim_Süreleri', 'İş_Yaşam_Dengesi', 'Şirkette_Geçen_Yıllar',
    'Mevcut_Roldeki_Yıllar', 'Son_Terfiden_Bu_Yana_Geçen_Yıllar', 'Mevcut_Yöneticideki_Yıllar'
]

# Alt grafiklerin yerleşimini belirleyin (6 sütunlu bir grid yapısı)
n_cols = 6
n_rows = (len(numerical_vars) + n_cols - 1) // n_cols

plt.figure(figsize=(20, n_rows * 4))

# Her bir sayısal değişken için histogram ve KDE grafikleri oluşturun
for i, var in enumerate(numerical_vars):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(df[var], kde=True)
    plt.title(f'{var} Dağılımı')
    plt.xlabel(var)
    plt.ylabel('Frekans')

plt.tight_layout()
plt.show()


- Yaş Dağılımı: Genç çalışan sayısı fazla, 30-35 yaş aralığında bir tepe noktası var. Yaş arttıkça frekans azalıyor.
- Günlük Ücret Dağılımı: Ücretler genellikle 500 ile 1000 arasında yoğunlaşıyor.
- Evden Uzaklık Dağılımı: Çoğu çalışan iş yerine yakın oturuyor (0-10 km).
- Eğitim Dağılımı: Eğitim seviyeleri arasında belirgin farklar var, çoğunlukla orta seviyelerde yoğunlaşmış.
- Çevre Memnuniyeti Dağılımı: Çevre memnuniyeti genel olarak yüksek, 4 ve 5 seviyelerinde yoğunlaşmış.
- Saatlik Ücret Dağılımı: Ücretler genel olarak 40-60 saat arasında yoğunlaşmış.
- İş Katılımı Dağılımı: İşe katılım düşük, bazı çalışanlar yüksek katılım gösteriyor.
- İş Seviyesi Dağılımı: Çoğu çalışan orta seviyelerde yer alıyor.
- İş Memnuniyeti Dağılımı: Memnuniyet genel olarak yüksek, 4 ve 5 seviyelerinde yoğunlaşmış.
- Maaş Artış Yüzdesi Dağılımı: Maaş artışları genellikle düşük oranlarda yoğunlaşmış.
- Performans Değerlendirmesi Dağılımı: Performans değerlendirmeleri genellikle 3-4 arasında yoğunlaşmış.
- İlişki Memnuniyeti Dağılımı: İlişki memnuniyeti genel olarak ortalama seviyelerde.
- İş Yaşam Dengesi Dağılımı: İş yaşam dengesi düşük seviyelerde yoğunlaşmış.
- Şirkette Geçen Yıllar Dağılımı: Çalışanlar genellikle 0-5 yıl arası şirkette çalışmış.
- Mevcut Roldeki Yıllar Dağılımı: Çoğu çalışan mevcut rolünde 0-5 yıl arasında çalışmış.
- Geçen Yıllık Eğitim Süreleri Dağılımı: Eğitim süreleri arasında büyük farklılıklar var, bazıları hiç eğitim almazken bazıları uzun süre eğitim almış.
- Hisse Senedi Opsiyon Seviyesi Dağılımı: Opsiyon seviyeleri arasında belirgin farklar var, çoğu çalışan orta seviyelerde yoğunlaşmış.
- Toplam Çalışma Yılları Dağılımı: Çalışma yılları genellikle 0-10 yıl arasında yoğunlaşmış.
- Son Terfiden Bu Yana Geçen Yıllar Dağılımı: Çoğu çalışan son terfisinden bu yana kısa süre geçmiş.
- Mevcut Yöneticiyle Yıllar Dağılımı: Çoğu çalışan mevcut yöneticisiyle 0-5 yıl arasında çalışmış.

df.columns

### İş Seviyesi ve İşten Ayrılma

# İş Seviyesi ve İşten Ayrılma
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='İşten_Ayrılma', y='İş_Seviyesi')
plt.title('İş Seviyesi ve İşten Ayrılma')
plt.xlabel('İşten Ayrılma')
plt.ylabel('İş Seviyesi')
plt.show()


- İşten ayrılan çalışanlar genellikle daha düşük iş seviyelerinde yoğunlaşmıştır (1-2). Bu, düşük iş seviyelerinin işten ayrılma riskini artırabileceğini gösterir.
- İşten ayrılmayan çalışanlar genellikle daha yüksek iş seviyelerinde (2-3) yoğunlaşmıştır.
- yüksek iş seviyelerindeki aykırı değerler bu seviyelerdeki (4-5)  bazı çalışanların işten ayrıldığını gösterir.

### Hipotez Testleri ve İstatistiksel Analiz

# İşten ayrılma grupları arasındaki maaş farkını test edelim
t_stat, p_val = stats.ttest_ind(df_cleaned[df_cleaned['İşten_Ayrılma'] == 1]['Aylık_Gelir'],
                                df_cleaned[df_cleaned['İşten_Ayrılma'] == 0]['Aylık_Gelir'])
print(f"T-İstatistiği: {t_stat}, P-Değeri: {p_val}")

T-İstatistiği ve P-Değeri Nedir?
- T-İstatistiği: İki grup arasındaki farkın büyüklüğünü gösterir. T-İstatistiği ne kadar büyükse (pozitif veya negatif), gruplar arasındaki fark o kadar anlamlıdır.
- P-Değeri: Bu farkın tesadüfi olarak meydana gelme olasılığını gösterir. P-Değeri ne kadar küçükse, farkın tesadüfi olmadığına dair güven o kadar yüksek olur.
İstatistiksel Anlamlılık:

P-Değeri çok küçüktür (0.000000000714736398535381), bu nedenle işten ayrılanlar ve ayrılmayanlar arasındaki maaş farkı istatistiksel olarak anlamlıdır. Bu farkın tesadüfen oluşma olasılığı son derece düşüktür.
- Yön ve Büyüklük:

Negatif T-İstatistiği (-6.203935765608938), işten ayrılanların ortalama maaşının işten ayrılmayanların ortalama maaşından daha düşük olduğunu gösterir. Bu, işten ayrılanların daha düşük maaş alıyor olabileceğini gösterir.

Rastgele Orman Modeli

# Hedef değişkeni belirleme ve veri önişleme
# 'İşten_Ayrılma' sütunu hedef değişkenimiz
df['İşten_Ayrılma'] = df['İşten_Ayrılma'].apply(lambda x: 1 if x == 'Yes' else 0)

# Gerekli sütunları seçme
features = ['Yaş', 'Günlük_Ücret', 'Evden_Uzaklık', 'Eğitim', 'Çevre_Memnuniyeti',
            'İş_Katılımı', 'İş_Seviyesi', 'İş_Memnuniyeti', 'Aylık_Gelir', 'Çalışılan_Şirket_Sayısı',
            'Maaş_Artış_Yüzdesi', 'Toplam_Çalışma_Yılları', 'Geçen_Yılki_Eğitim_Süreleri', 'Şirkette_Geçen_Yıllar']

X = df[features]
y = df['İşten_Ayrılma']

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele orman modeli oluşturma
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_rf = rf_model.predict(X_test)

# Modeli değerlendirme
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))


Modelimizin doğruluğu %87 civarında, bu oldukça iyi bir sonuç. Ancak, modelin işten ayrılma (1 sınıfı) durumlarını tespit etme yeteneği düşük. Precision, recall ve f1-score gibi metrikler, modelin bu sınıftaki performansının iyileştirilmesi gerektiğini gösteriyor.

#### Modeli İyileştirme

Modelin performansını iyileştirmek için hiperparametre optimizasyonu yapabilirsiniz. Örneğin, GridSearchCV kullanarak en iyi parametreleri bulabilirsiniz:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Orijinal veri setinizi yükleyin veya tanımlayın
# X ve y değişkenleri burada orijinal özellikler ve hedef etiketler olmalıdır

# Veri dengesizliğini gidermek için SMOTE uygulama
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# GradientBoostingClassifier modelini oluşturma
gbc_model = GradientBoostingClassifier(random_state=42)

# Hiperparametreleri belirleme
param_grid_gbc = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0]
}

# GridSearchCV uygulama
grid_search_gbc = GridSearchCV(estimator=gbc_model, param_grid=param_grid_gbc, cv=5, n_jobs=-1, verbose=2)
grid_search_gbc.fit(X_res, y_res)

# En iyi parametreler ile model oluşturma
best_gbc_model = grid_search_gbc.best_estimator_
best_gbc_model.fit(X_res, y_res)

# Test veri seti üzerinde tahmin yapma
y_pred_gbc = best_gbc_model.predict(X_test)

# Modeli değerlendirme
print("Gradient Boosting Classifier Accuracy:", accuracy_score(y_test, y_pred_gbc))
print("Gradient Boosting Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gbc))
print("Gradient Boosting Classifier Classification Report:\n", classification_report(y_test, y_pred_gbc))


Modeli Eğitme ve En İyi Modeli Bulma:

# Hiperparametreleri belirleme
param_grid_gbc = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0]
}

# GradientBoostingClassifier modeli
gbc_model = GradientBoostingClassifier(random_state=42)

# GridSearchCV uygulama
grid_search_gbc = GridSearchCV(estimator=gbc_model, param_grid=param_grid_gbc, cv=5, n_jobs=-1, verbose=2)
grid_search_gbc.fit(X_res, y_res)

# En iyi parametreler ile model oluşturma
best_gbc_model = grid_search_gbc.best_estimator_


GridSearchCV ile En İyi Modeli Bulma:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Parametre ızgarası
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0]
}

# Gradient Boosting Classifier modelini oluşturma
gbc = GradientBoostingClassifier()

# GridSearchCV
grid_search_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# GridSearchCV ile modeli eğitme
grid_search_gbc.fit(X_train, y_train)


# GridSearchCV ile modeli eğitme
grid_search_gbc.fit(X_train, y_train)


En İyi Modeli Bulma ve Tahmin Yapma:



# En iyi parametreler ile model oluşturma
best_gbc_model = grid_search_gbc.best_estimator_

# Tahmin yapma
y_pred_gbc = best_gbc_model.predict(X_test)



Modeli Değerlendirme

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Modeli değerlendirme
print("Gradient Boosting Classifier Accuracy:", accuracy_score(y_test, y_pred_gbc))
print("Gradient Boosting Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gbc))
print("Gradient Boosting Classifier Classification Report:\n", classification_report(y_test, y_pred_gbc))

# En iyi parametreleri yazdırma
print("En iyi parametreler:", grid_search_gbc.best_params_)




#### Gradient Boosting Classifier Sonuçları

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# En iyi model ile tahmin yapma
y_pred_gbc = best_gbc_model.predict(X_test)

# Sonuçları değerlendirme
print("Gradient Boosting Classifier Accuracy:", accuracy_score(y_test, y_pred_gbc))
print("Gradient Boosting Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gbc))
print("Gradient Boosting Classifier Classification Report:\n", classification_report(y_test, y_pred_gbc))


Model Performansı:

Accuracy (Doğruluk): 0.8775510204081632

Doğruluk, tüm doğru tahminlerin toplam tahminlere oranını gösterir. Yani, modelinizin genel doğruluğu %87.76'dır. Bu, modelinizin oldukça iyi performans gösterdiğini gösterir.
Confusion Matrix (Karışıklık Matrisi):
[[250   5]
 [ 31   8]]
Modeliniz, 250 doğru negatif (gerçekten işten ayrılmayanları doğru tahmin etmiş) ve 8 doğru pozitif (gerçekten işten ayrılanları doğru tahmin etmiş) tahmin yapmış.
Yanlış negatif (gerçekten işten ayrılanları yanlış tahmin etmiş) sayısı 31 ve yanlış pozitif (gerçekten işten ayrılmayanları yanlış tahmin etmiş) sayısı 5'tir.

Classification Report (Sınıflandırma Raporu):
              precision    recall  f1-score   support

         0       0.89      0.98      0.93       255
         1       0.62      0.21      0.31        39

  accuracy                           0.88       294
 macro avg       0.75      0.59      0.62       294
weighted avg 0.85 0.88 0.85 294
- **Precision (Kesinlik)**: Pozitif tahminlerin ne kadarının doğru olduğunu gösterir. "1" sınıfı için precision değeri 0.62'dir, yani model işten ayrılmayı tahmin ettiğinde %62 doğru tahmin yapmış.
- **Recall (Duyarlılık)**: Gerçek pozitiflerin ne kadarının doğru tahmin edildiğini gösterir. "1" sınıfı için recall değeri 0.21'dir, yani gerçek işten ayrılmaların %21'i doğru tahmin edilmiştir.
- **F1-Score**: Precision ve recall'un harmonik ortalamasıdır. "1" sınıfı için F1-score değeri 0.31'dir.
- **Macro Average**: Her iki sınıfın ortalamasıdır ve dengeli veri setlerinde kullanışlıdır. F1-score için macro avg değeri 0.62'dir.
- **Weighted Average**: Her iki sınıfın ortalamasıdır ve sınıf dağılımını dikkate alır. F1-score için weighted avg değeri 0.85'tir.

### Değerlendirme:

- Modelinizin doğruluğu yüksek (%87.76), bu da modelin genel olarak iyi performans gösterdiğini gösteriyor.
- Ancak, sınıf dengesizliği nedeniyle "1" sınıfı için recall ve F1-score değerleri düşük. Bu, modelinizin işten ayrılmaları tespit etmede zorlandığını gösterir.
- Daha dengeli bir sınıf dağılımı sağlamak veya SMOTE gibi yöntemler kullanarak veri setini dengelemek, model performansını artırabilir.

Sonuç olarak, modeliniz genel olarak iyi performans gösteriyor, ancak işten ayrılmaları tespit etme konusunda iyileştirme yapmanız faydalı olabilir. Özellikle düşük recall ve F1-score değerleri, işten ayrılmaları kaçırma riskinizin yüksek olduğunu gösteriyor. Bu sonuçlarla ilgili daha fazla iyileştirme yapmak isterseniz, veri dengesizliğini ele almak iyi bir adım olabilir.


### XGBoost (Extreme Gradient Boosting) Modeli

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# XGBoost model nesnesi oluşturma
xgb_model = xgb.XGBClassifier(random_state=42)

# Hiperparametreler için grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

# GridSearchCV ile en iyi parametreleri bulma
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# En iyi modeli alma
best_xgb_model = grid_search_xgb.best_estimator_

# Tahmin yapma
y_pred_xgb = best_xgb_model.predict(X_test)

# Modeli değerlendirme
print("XGBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("XGBoost Classifier Classification Report:\n", classification_report(y_test, y_pred_xgb))

# En iyi parametreler
print("En iyi parametreler:", grid_search_xgb.best_params_)


 XGBoost, özellikle sınıflandırma ve regresyon problemlerinde yüksek performans gösteren bir gradient boosting algoritmasıdır.

- Random Forest Classifier Sonuçları:Accuracy (Doğruluk): 0.8333333333333334
- Gradient Boosting Classifier Sonuçları :Accuracy (Doğruluk): 0.8775510204081632
- XGBoost Classifier Sonuçları:Accuracy (Doğruluk): 0.87

- Sonuç
Gradient Boosting Classifier en iyi genel performansı gösteriyor. Ancak, XGBoost ve Random Forest da iyi sonuçlar vermektedir. Uygulamada, model seçimi yaparken doğruluk, duyarlılık ve f1-score gibi metrikler dikkate alınmalıdır. Ayrıca, modelin açıklanabilirliği, eğitim süresi ve tahmin süresi gibi diğer faktörler de göz önünde bulundurulmalıdır.







