############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################
#Veri içerisindeki örüntüleri (pattern, ilişki, yapı) bulmak için kullanılan kural tabanlı bir makine öğrenmesi tekniğidir.
#Apriori Algoritması: Sepet analizi yöntemidir. Ürünlerin birlikteliklerini ortaya çıkarmak için kullanılır.

## İŞ PROBLEMİ
# Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en uygun ürün önerisini birliktelik kuralı kullanarak yapınız.
# Ürün önerileri 1 tane ya da 1'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz.
# Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
# Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
# Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747

## VERİ SETİ HİKAYESİ
# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış
# işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

# Invoice      : Fatura Numarası (Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder)
# StockCode    : Ürün kodu (Her bir ürün için eşsiz)
# Description  : Ürün ismi
# Quantity     : Ürün adedi (Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate  : Fatura tarihi
# UnitPrice    : Fatura fiyatı (Sterlin)
# CustomerID   : Eşsiz müşteri numarası
# Country      : Ülke ismi
# 8 Değişken 541.909 Gözlem 45.6MB

## UYGULANACAK ADIMLAR
# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
# 5. Çalışmanın Scriptini Hazırlama

############################################
# 1. Veri Ön İşleme
############################################
#Görev 1: Veriyi Hazırlama
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()

# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.

df.head()
df.tail()
df.describe().T #İade işlemleri C ile işlendiğinden Quantity'lerde eksi değerler gelmektedir ve Outlier problemi mevcut
df.shape
df.isnull().sum() #Description ve Customer ID için eksik değerler mevcut olduğu görüldü.
cancel_rows = df[df['Invoice'].str.startswith('C', na=False)]
print(cancel_rows)  #9288  adet iptal fatura mevcut. Bunlar eksi değerlere sebep olmakta.

# Adım 2-3-4-5 çözümü:
def retail_data_prep(dataframe):
    # adım 2. StockCode'u 'POST' olanları çıkar.
    dataframe = dataframe[dataframe["StockCode"] != "POST"].copy()
    # adım 3. Eksik değerleri sil
    dataframe.dropna(inplace=True)
    # adım 4. Invoice içerisinde C bulunan değerleri veri setinden çıkar (iptaller)
    dataframe = dataframe[~dataframe["Invoice"].str.contains('C', na=False)]
    # adım 5. Quantity ve Price pozitif olanları al
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)


# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe["Invoice"] = dataframe["Invoice"].astype(str)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

df.isnull().sum()
df.describe().T

###########################################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
##########################################################
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
# Adım 1: Fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.

    #istenen veri yapısı görüntüsü aşağıdaki gibi olmalı
        # Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
        # Invoice
        # 536370                              0                                 1                       0
        # 536852                              1                                 0                       1
        # 536974                              0                                 0                       0
        # 537065                              1                                 0                       0
        # 537463                              0                                 0                       1


df_germany = df[df["Country"] == "Germany"]
df_germany.head()

df_germany.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

germany_inv_pro_df = create_invoice_product_df(df_germany)
germany_inv_pro_df = create_invoice_product_df(df_germany, id=True)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.
def create_rules(dataframe, id=False, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country] 
    dataframe = create_invoice_product_df(dataframe, id=False)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True) 
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01) 
    return rules #ve bunu return et

rules = create_rules(df, id=True, country="Germany")

print(create_rules(rules, 21987))
print(create_rules(rules, 23235))
print(create_rules(rules, 22747))

################################################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
################################################################
# Görev 3: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_germany, 21987)  # Ürün: ['PACK OF 6 SKULL PAPER CUPS']
check_id(df_germany, 23235)  # Ürün: ['STORAGE TIN VINTAGE LEAF']
check_id(df_germany, 22747)  # Ürün: ["POPPY'S PLAYHOUSE BATHROOM"]


# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

product_name = 'PACK OF 6 SKULL PAPER CUPS'
check_id(df, product_name)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, antecedents_set in enumerate(sorted_rules["antecedents"]):
    # antecedents bir set, product_id içeriyor mu kontrol et
    if product_name in antecedents_set:
        consequents_set = sorted_rules.iloc[i]["consequents"]
        # consequents seti boş değilse ekle
        if len(consequents_set) > 0:
            # consequents içindeki tüm ürünleri ekle
            for item in consequents_set:
                # Aynı ürünü önerme
                if item != product_name and item not in recommendation_list:
                    recommendation_list.append(item)

print(recommendation_list)


def arl_recommender(rules_df, product_name, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, antecedents_set in enumerate(sorted_rules["antecedents"]):
        # antecedents bir set, product_id içeriyor mu kontrol et
        if product_name in antecedents_set:
            consequents_set = sorted_rules.iloc[i]["consequents"]
            # consequents seti boş değilse ekle
            if len(consequents_set) > 0:
                # consequents içindeki tüm ürünleri ekle
                for item in consequents_set:
                    # Aynı ürünü önerme
                    if item != product_name and item not in recommendation_list:
                        recommendation_list.append(item)
    return recommendation_list[0:rec_count]

arl_recommender(rules, 'PACK OF 6 SKULL PAPER CUPS', 3)
arl_recommender(rules, 'STORAGE TIN VINTAGE LEAF', 3)
arl_recommender(rules, "POPPY'S PLAYHOUSE BATHROOM", 3)

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

products = [
    'PACK OF 6 SKULL PAPER CUPS',
    'STORAGE TIN VINTAGE LEAF',
    "POPPY'S PLAYHOUSE BATHROOM"]

for prod in products:
    recs = arl_recommender(rules, prod, 3)
    print(f"'{prod}' ürünü için öneriler: {recs if recs else 'Öneri bulunamadı.'}")

### ÖNERİLER : ###
# 'PACK OF 6 SKULL PAPER CUPS' ürünü için öneriler: ['PACK OF 20 SKULL PAPER NAPKINS', 'SET/6 RED SPOTTY PAPER CUPS', 'SET/6 RED SPOTTY PAPER PLATES']
# 'STORAGE TIN VINTAGE LEAF' ürünü için öneriler: ['SET OF 4 KNICK KNACK TINS DOILY ', 'ROUND STORAGE TIN VINTAGE LEAF', 'SET OF TEA COFFEE SUGAR TINS PANTRY']
# 'POPPY'S PLAYHOUSE BATHROOM' ürünü için öneriler: ["POPPY'S PLAYHOUSE BEDROOM ", "POPPY'S PLAYHOUSE LIVINGROOM ", "POPPY'S PLAYHOUSE KITCHEN"]


############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe["Invoice"] = dataframe["Invoice"].astype(str)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=False, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id=False)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \

sort_values("confidence", ascending=False)
