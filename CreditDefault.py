# Manupulasyon Libraryleri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# gc.collect
import gc

# Encoderlar, Scalerlar ve ML modelleri Import etme.
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# PIPELINE
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
# RANDOM SAMPLING
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV
from sklearn.utils import resample
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE

# Satir- Sutun Boyutlari ayarlama.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Column tiplerini belirleme
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = [col for col in cat_cols if col not in "TARGET"]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
    num_cols = [col for col in num_cols if col not in "index"]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

## Train setine df, test setine apptest dedim.
apptrain_ = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
apptrain = apptrain_.copy()
apptest_ = pd.read_csv("../input/home-credit-default-risk/application_test.csv")
apptest = apptest_.copy()

df = pd.concat([apptrain.assign(ind="train"),apptest.assign(ind="test")])

# gc.collect kullandım
cat_cols, num_cols, cat_but_car = grab_col_names(df)
def feature_func1(dataframe):
    dataframe = dataframe.loc[dataframe['CODE_GENDER'] != 'XNA']
    dataframe['DAYS_BIRTH'] = dataframe['DAYS_BIRTH'].abs()/-365
    dataframe['DAYS_EMPLOYED'].replace(365243,np.nan, inplace=True)
    dataframe['INCOME_PER_PERSON'] = dataframe['AMT_INCOME_TOTAL'] / dataframe['CNT_FAM_MEMBERS']
    dataframe["INCOME_PER_CHILD"] = dataframe["AMT_INCOME_TOTAL"] / dataframe["CNT_CHILDREN"]
    dataframe["INCOME_PER_CHILD"] = dataframe["AMT_INCOME_TOTAL"] / (1 + dataframe["CNT_CHILDREN"])
    dataframe['MaasVUrunFiyat'] = dataframe['AMT_INCOME_TOTAL']/ dataframe['AMT_GOODS_PRICE']
    dataframe['Age_cat'] = pd.cut(x = dataframe['DAYS_BIRTH'], bins = [19, 25, 41, 50, 70], labels = ['Young', 'Adult', 'Mature', 'Retirement'])
    dataframe['MaasVsKrediTutari'] =dataframe['AMT_INCOME_TOTAL']/ dataframe['AMT_CREDIT']
    dataframe['KrediTutariVsUrunFiyati'] =dataframe['AMT_CREDIT'] - dataframe['AMT_GOODS_PRICE']
    dataframe['MusteriSkor'] = dataframe['REGION_RATING_CLIENT']*dataframe['REGION_RATING_CLIENT_W_CITY']
    dataframe["EMPLOYED_BIRTH_DAYS"] = dataframe["DAYS_EMPLOYED"] / dataframe["DAYS_BIRTH"]
    dataframe['NEW_SOURCES_PROD'] = dataframe['EXT_SOURCE_1'] * dataframe['EXT_SOURCE_2'] * dataframe['EXT_SOURCE_3']
    dataframe['NEW_EXT_SOURCES_MEAN'] = dataframe[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    dataframe['NEW_SCORES_STD'] = dataframe[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    dataframe["DAYS_LAST_PHONE_CHANGE"] = dataframe["DAYS_LAST_PHONE_CHANGE"]/-1
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        dataframe[bin_feature], uniques = pd.factorize(dataframe[bin_feature])
    #dataframe['NEW_CREDIT_TO_INCOME_RATIO'] = dataframe['AMT_CREDIT'] / dataframe['AMT_INCOME_TOTAL']
    dataframe['Industry'] = dataframe['ORGANIZATION_TYPE'].apply(lambda x: x if 'Industry' in x else 'NotIndustry')
    dataframe['Business Entity'] = dataframe['ORGANIZATION_TYPE'].apply(lambda x: x if 'Business Entity' in x else 'NotBsiness')
    dataframe['Trade'] = dataframe['ORGANIZATION_TYPE'].apply(lambda x: x if 'Trade' in x else 'NotTrade')
    dataframe['Transport'] = dataframe['ORGANIZATION_TYPE'].apply(lambda x: x if 'Transport' in x else 'NotTransport')
    dataframe, cat_cols = one_hot_encoder(dataframe, nan_as_category=False)
    return dataframe
df1 = feature_func1(df)

gc.collect()

#### credit_card_balance kart için ön hazırlıkları yaptım. tüm kolonları alıp bunların aggragationlarla aldım


# One-hot-encoder icerde kullaniliyor.
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



### credit_card_balance  prep
def credit_card_prep(dataframe):
    cred = pd.read_csv("../input/home-credit-default-risk/credit_card_balance.csv")
    cat_cols, num_cols, cat_but_car = grab_col_names(cred)
    cred, cat_cols = one_hot_encoder(cred, nan_as_category= True)
    cred.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cred_agg = cred.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cred_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cred_agg.columns.tolist()])
    cred_agg['CC_COUNT'] = cred.groupby('SK_ID_CURR').size()
    dataframe= dataframe.join(cred_agg, how='left', on='SK_ID_CURR')
    return dataframe

df2 = credit_card_prep(df1)

gc.collect()

#### pos_cash için

def pos_feature(dataframe):
    pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
    # pos' taki sutun tipleri
    cat_cols, num_cols, cat_but_car = grab_col_names(pos)

    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    pos_agg = pos.groupby("SK_ID_CURR").agg({'MONTHS_BALANCE': ['max', 'mean', 'size'],
                                          'SK_DPD': ['max', 'mean'],
                                           'SK_DPD_DEF': ['max', 'mean']})
    # Agrigation daki birinci ve ikinci bölümlerin isimlerini büyüterek kolona ekledim
    pos_agg.columns = pd.Index(['POS' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # işimize yarayacak dataframe pos_agg oldu. pos dataframe'ini çalışmadan çık
    del pos
    ## pos_agg yi ana veriye ekledim.
    dataframe = dataframe.join(pos_agg, how='left', on='SK_ID_CURR')
    return dataframe

df3 = pos_feature(df2)

gc.collect()


def inst_feature(dataframe):
    inst = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')

    # pos' taki sutun tipleri
    cat_cols, num_cols, cat_but_car = grab_col_names(inst)

    inst, cat_cols = one_hot_encoder(inst, nan_as_category=True)
    inst["PAYMENT_RATE"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    # taksit ödemesi - ödenen ücret
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

    # V_G vadesi geçen gün sayısı, V_O Son ödemesine kalan gün sayısı
    inst['V_G'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['V_O'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']

    # Negatif değer yerine optimize etmiş gibi sıfır koysak?
    inst['V_G'] = inst['V_G'].apply(lambda x: x if x > 0 else 0)
    inst['V_O'] = inst['V_O'].apply(lambda x: x if x > 0 else 0)

    # Her bir taksitte ödenen miktar
    inst["PER_PAYMENT_NUMBER"] = inst["AMT_PAYMENT"] / inst["NUM_INSTALMENT_NUMBER"]

    ### Taksit sayısıyla, vadesinden önce ödeme oranı arasında negatif bir korelasyon var.
    # Taksit sayısı arttıkça, vadesinden önce ödeme düşüyor.

    # Aynı işlemleri inst için de yaptım ve inst dataframe silebiliriz.
    inst_agg = inst.groupby("SK_ID_CURR").agg({'PER_PAYMENT_NUMBER': ['max', 'mean', 'size'],
                                               'V_O': ['max', 'mean'],
                                               'V_G': ['max', 'mean'],
                                               'PAYMENT_DIFF': ['max', 'mean'],
                                               'PAYMENT_RATE': ['max', 'mean'],
                                               })
    inst_agg.columns = pd.Index(['INST' + e[0] + "_" + e[1].upper() for e in inst_agg.columns.tolist()])
    del inst
    dataframe = dataframe.join(inst_agg, on='SK_ID_CURR', how='left')
    return dataframe


df4 = inst_feature(df3)

gc.collect()


def feature_func2(dataframe):
    dataframe["GOOD_EXIT_MEAN_DIVIDE"] = dataframe["AMT_GOODS_PRICE"] / dataframe["NEW_EXT_SOURCES_MEAN"]
    dataframe["GOOD_EXIT_MEAN_CROSS"] = dataframe["AMT_GOODS_PRICE"] * dataframe["NEW_EXT_SOURCES_MEAN"]
    dataframe["EXIT_BIRTH_DIVIDE"] = dataframe["DAYS_BIRTH"] / dataframe["EXT_SOURCE_3"]
    dataframe["EXIT_BIRTH_CROS"] = dataframe["DAYS_BIRTH"] * dataframe["EXT_SOURCE_3"]
    dataframe["GOOD_EXIT_MEAN_DIVIDE_BIRTH_CROS"] = dataframe["GOOD_EXIT_MEAN_DIVIDE"] / dataframe["EXT_SOURCE_3"]
    dataframe["DAYS_EXT_DIVIDE"] = dataframe["DAYS_BIRTH"] / dataframe["NEW_EXT_SOURCES_MEAN"]
    dataframe["DAYS_EXT_CROSS"] = dataframe["DAYS_BIRTH"] * dataframe["NEW_EXT_SOURCES_MEAN"]
    dataframe["DAYS_SOURCE_2_DIVIDE"] = dataframe["DAYS_ID_PUBLISH"] / dataframe["EXT_SOURCE_2"]
    dataframe["DAYS_SOURCE_2_CROSS"] = dataframe["DAYS_ID_PUBLISH"] * dataframe["EXT_SOURCE_2"]
    dataframe["G_E_DIVIDE"] = dataframe["GOOD_EXIT_MEAN_DIVIDE"] / dataframe["EXIT_BIRTH_DIVIDE"]
    dataframe["D_D_CROSS"] = dataframe["DAYS_EXT_CROSS"] * dataframe["DAYS_SOURCE_2_DIVIDE"]
    dataframe["D_D_DIVIDE"] = dataframe["DAYS_EXT_DIVIDE"] / dataframe["DAYS_SOURCE_2_DIVIDE"]
    #dataframe["1"] = dataframe["DAYS_EXT_DIVIDE"] / dataframe["GOOD_EXIT_MEAN_DIVIDE"]
    return dataframe

df5 = feature_func2(df4)

# Categoric - Numeric Column UPDATE before OHE.
cat_cols, num_cols, cat_but_car = grab_col_names(df5)

# Sutun isimlerini python ML algoritmasi icin uygun hale getirme
import re
df7 = df6.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

del df7["Age_cat"]

train_df = df7[df7['TARGET'].notnull()]
test_df = df7[df7['TARGET'].isnull()]

y = train_df["TARGET"]
X = train_df.drop(["TARGET",'SK_ID_CURR'], axis=1)

"""f,ax=plt.subplots(1,2,figsize=(18,8))
df['TARGET'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('dağılım')
ax[0].set_ylabel('')
sns.countplot('TARGET',data=df,ax=ax[1])
ax[1].set_title('TARGET')
plt.show()"""

from imblearn.under_sampling import RandomUnderSampler
# transform the dataset
ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X, y)

# resampling

lgbm_base = LGBMClassifier(random_state=46).fit(X_ranUnSample, y_ranUnSample)

cv_results = cross_validate(lgbm_base, X_ranUnSample, y_ranUnSample, cv=5, scoring=["roc_auc"])

cv_results['test_roc_auc'].mean()

lgbm_final_model = LGBMClassifier(random_state=17, max_bin=255, n_estimators=12000, colsample_bytree=0.9, learning_rate=0.01).fit(X_ranUnSample, y_ranUnSample)

cv_results = cross_validate(lgbm_final_model, X_ranUnSample, y_ranUnSample, cv=5, scoring=["roc_auc"])

cv_results['test_roc_auc'].mean()

def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr)
    plt.show()
    pass

#generate_auc_roc_curve(lgbm_model, X_test)

y_t = test_df['TARGET']
X_t = test_df.drop(["TARGET",'SK_ID_CURR'], axis=1)

# Bu sonradan silinebilir.
X_t.to_csv('mycsvfile3.csv',index=False)

y_pred= lgbm_final_model.predict(X_t)

my_submission = pd.DataFrame({'SK_ID_CURR': test_df.SK_ID_CURR, 'TARGET': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission14.csv', index=False)

import joblib

# Bu sonuclari cikarmak icin
joblib.dump(lgbm_final_model, 'lgbm_final.pkl')
