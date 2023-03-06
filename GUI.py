import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
from importlib import reload
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

import pyspark
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, NaiveBayesModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from underthesea import word_tokenize, pos_tag, sent_tokenize
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string

from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.sql.functions import col, lit
from pyspark.ml import PipelineModel
import pyspark.sql.functions as F
from pyspark.sql.functions import *
import regex
import io

from pyspark.mllib.evaluation import MulticlassMetrics
import warnings
import os
import base64
from pathlib import Path
from st_aggrid import AgGrid,GridUpdateMode,DataReturnMode, JsCode,ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder



import findspark
findspark.init()

warnings.filterwarnings("ignore")

# header=st.container()
# dataset=st.container()
# features=st.container()
# modelTraining=st.container()



############################################LOAD EMOJICON
file = open('Data/files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('Data/files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('Data/files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('Data/files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('Data/files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='600'>".format(
    img_to_bytes(img_path)
    )
    return img_html

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words   
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '                    
    document = new_sentence  
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

# Chuẩn hóa unicode tiếng việt
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def convert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
    
# có thể bổ sung thêm các từ: chẳng, chả...
# có thể bổ sung thêm các từ: chẳng, chả...
# có thể bổ sung thêm các từ: chẳng, chả...
def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                    #print(word)
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    elif 'tạm' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
#             print(word)
#             print(i)
            if  word == 'tạm':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]

                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
            #print(new_text)
    elif 'hơi' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'hơi':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
#                     print(word)
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
            #print(new_text)
#     elif 'chất_lượng' in text_lst:
#         while i <= len(text_lst) - 1:
#             word = text_lst[i]
#             if  word == 'chất_lượng':
#                 next_idx = i+1
#                 if next_idx <= len(text_lst) -1:
#                     word = word +'_'+ text_lst[next_idx]
#                 i= next_idx + 1
#             else:
#                 i = i+1
#             new_text = new_text + word + ' '
            #print(new_text)
    elif 'quá' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'quá':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
            #print(new_text)
    elif 'giao' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'giao':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
            #print(new_text)
    elif 'khá' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'khá':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
            #print(new_text)
    elif 'hơn' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            if  word == 'hơn':
                back_idx = i-1
                if back_idx >=1:
                    cache=text_lst[back_idx]
                    new_text=new_text.replace(cache, '')
#                     print(new_text)
#                     print(text_lst)
                    word =  cache+'_'+ word
                    #print(text_lst)
                i= back_idx + 2
            else:
                i = i+1
            #print(word)
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

def xu_li_text_1(word,lst_word_type,lst_word_type_special):
    if (word[1].upper() in lst_word_type) and word[0]!="chất_lượng":
        return word[0]
    elif (word[1].upper() in lst_word_type_special) and word[0]=="chất_lượng":
        return word[0]
    else:
        return ""


def func_class(s):
    if s>3:
        return "Like"
    elif s<=2:
        return "Not_Like"
    return "Neutral"


def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['A','AB','VB','VY','R','M','Nu','V','N','NP']
        lst_word_type_special= ['A','AB','VB','VY','R','M','Nu']
        #lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join(xu_li_text_1(word,lst_word_type,lst_word_type_special) for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document


def xu_li_text(text):
    document=  process_text(text,emoji_dict,teen_dict,wrong_lst)
    document = convert_unicode(document)
    document = process_postag_thesea(document)
    document = remove_stopword(document, stopwords_lst)
    return document

def xu_li_text_dataframe(data):
    pre_data_lst=[]
    for row in range(len(data)):
        document = data.iloc[row]["comment"]
        document=  process_text(document,emoji_dict,teen_dict,wrong_lst)
        document = convert_unicode(document)
        document = process_postag_thesea(document)
        document = remove_stopword(document, stopwords_lst)
        pre_data_lst.append(document)
    data['pre_comment'] = pre_data_lst
    return data



def cluster_function(prediction):
    if prediction =="Cluster 0":
        return "Regular"
    elif prediction =="Cluster 1":
        return "Loyal"
    elif prediction =="Cluster 2":
        return "Star"
    elif prediction =="Cluster 3":
        return "Big Spender"
    return "Lost Cheap"

## Function to check skewness
def check_skew(df_skew, column):
    """
    The function takes a dataframe and a column name as input, and returns a plot of the distribution of
    the column, along with the skew and the p-value of the skewtest
    
    :param df_skew: the dataframe you want to check
    :param column: The column name of the dataframe you want to check the skew of
    :return: The skewness of the data and the p-value of the skewness test.
    """
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return



# ----------- Global Sidebar ---------------

condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA","Preprocessing","Model Evaluation", "Model Prediction")
)

# ------------- Introduction ------------------------

if condition == 'Introduction':
    #st.image(os.path.join(os.path.abspath(''), 'data', 'dataset-cover.jpg'))
    st.subheader('About')
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides an overview of the brazilian_houses_to_rent dataset from Kaggle. It is a dataset that provides rent prices for real estate properties in Brazil.
    The data were provided from this [source](https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent). 
    You can check on the sidebar:
    - EDA (Exploratory Data Analysis)
    - Model Prediction
    - Model Evaluation
    The prediction are made regarding to the rent amount utilizing pre trained machine learning models.
    All the operations in the dataset were already done and stored as csv files inside the data directory. If you want to check the code, go through the notebook directory in the [github repository](https://github.com/arturlunardi/predict_rental_prices_streamlit).
    """)

    st.subheader('Model Definition')

    st.write("""
    The structure of the training it is to wrap the process around a scikit-learn Pipeline. There were 4 possible combinations and 5 models, resulting in 20 trained models.
    The combinations are regarding to perform Feature Creation and/or Target Transformations in the dataset.
    Models:
    - Random Forest
    - XGB
    - Ridge
    - LGBM
    - Neural Network
    Our main accuracy metric is RMSE. To enhance our model definition, we utilized Cross Validation and Random Search for hyperparameter tuning.
    """)
# ------------- EDA ------------------------

elif condition == 'EDA':

    data = pd.read_csv("http://dramas.pro/Products_Shopee_comments.csv", delimiter=',')
    original_title_data = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Raw Data</b></p>'
    st.markdown(original_title_data,unsafe_allow_html=True)
    st.dataframe(data.head(20))

    cols=data.category.unique()
    count_data=data.groupby("category")[["product_id","comment"]].count()
    count_data.columns=["number of rows","number of not null comment"]
    original_title_data = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>The number of each category and the number of comments on them</b></p>'
    st.markdown(original_title_data,unsafe_allow_html=True)
    st.dataframe(count_data)

    
    sub_data = data[(data["category"]=="Thời Trang Nam")][["comment","rating"]]
    original_title_visual_products = '<p style="font-family:Garamond, serif; color:black; font-size: 20px;"><b> &#9830; To run the model, select the comment and rating columns</b></p>'
    st.markdown(original_title_visual_products,unsafe_allow_html=True)
    st.dataframe(sub_data)


    buffer = io.StringIO()
    sub_data.info(buf=buffer)
    info_subdata = buffer.getvalue()
    original_title_visual_products = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Print out info of dataset</b></p>'
    st.markdown(original_title_visual_products,unsafe_allow_html=True)
    st.text(info_subdata)

elif condition == 'Preprocessing':
    data = pd.read_csv("https://dramas.pro/Products_Shopee_comments.csv", delimiter=',')

    original_title_visual_products = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b> Data before Preprocessing</b></p>'
    st.markdown(original_title_visual_products,unsafe_allow_html=True)
    sub_data = data[data["category"]=="Thời Trang Nam"][["comment","rating"]]
    st.dataframe(sub_data)

    original_title_visual_products = '<p style="font-family:Garamond, serif; color:black; font-size: 20px;"><b> &#9830; To run the model, select the comment and rating columns</b></p>'
    st.markdown(original_title_visual_products,unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Check Null</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(30, 103, 119);
        overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: blue;
        }

        [data-testid="stMetricValue"] {
            font-size: 30px;
        }
        </style>
        """
    , unsafe_allow_html=True)

    data_check_null=sub_data.isnull().sum().to_frame('counts')
   
    
    col1.metric("Total number of null columns",str(data_check_null.loc[data_check_null["counts"]>0].columns.size))


    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Black; font-size: 20px;"><b> &#9830; Show null values</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.dataframe(data_check_null)

    sub_data=sub_data.dropna()

    sub_data["class"]=sub_data["rating"].apply(lambda x: func_class(x))

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Data after removing null and applying class</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.dataframe(sub_data)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>WordCloud before text cleaning</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('Data/wordcloud_prefix.png')+"</p>", unsafe_allow_html=True)

    original_title_visual_products = '<p style="font-family:Garamond, serif; color:blue; font-size: 30px;"><b>Using text Processing Library to Clean Text</b></p>'
    st.markdown(original_title_visual_products,unsafe_allow_html=True)

    code_lib_clean_text_1 ='''pre_data_lst=[]
for row in range(len(sub_data)):
    document = sub_data.iloc[row]["comment"]
    document=  process_text(document,emoji_dict,teen_dict,wrong_lst)
    document = convert_unicode(document)
    document = process_postag_thesea(document)
    document = remove_stopword(document, stopwords_lst)
    pre_data_lst.append(document)
    '''
    st.code(code_lib_clean_text_1, language ='python')

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Descriptive statistic of word token</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)

    data_pre = pd.read_csv("Data/pre_data_7.csv", delimiter=',',index_col=0)
    data_resample = pd.read_csv("Data/data_resample.csv", delimiter=',',index_col=0)

    data_pre["count_word"]=data_pre["pre_comment"].apply(lambda x:len(x.split()))
    counts_word=data_pre.groupby("class")["count_word"].agg(["min","max"])
    st.dataframe(counts_word)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>WordCloud after cleaning</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('Data/wordcloud_preprocessing.png')+"</p>", unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Black; font-size: 25px;"><b>&#8594 After cleaning up, we saw that each class\'s keyword was clear and representative.</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 25px;"><b>The data is imbalanced, with the Like class outnumbering the Neutral and Not Like classes by a factor of ten.</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)

    
    #count_=data_pre.groupby("class").size().to_frame('counts')
    st.write(data_pre["class"].value_counts())

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 25px;"><b>Data after resampling</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.write(data_resample["class"].value_counts())




elif condition == 'Model Evaluation':
 
    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>BernoulliNB</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('Data/BernoulliNB_eval.png')+"</p>", unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 25px;"><b>LogisticRegression</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('Data/LogisticRegression_eval.png')+"</p>", unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 25px;"><b>DecisionTree</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('Data/DecisionTree_eval.png')+"</p>", unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 25px;"><b>ExtraTreeClassifier</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('Data/ExtraTreeClassifier_eval.png')+"</p>", unsafe_allow_html=True)

    original_title_duplicate = '<p style="font-family:Garamond, serif; color:Black; font-size: 25px;"><b>&#8594 DecisionTree and ExtraTree Models are good enough for Sentiment Analysis because of their accuracy, f1-scores, and recall of over 90%.</b></p>'
    st.markdown(original_title_duplicate,unsafe_allow_html=True)


elif condition == 'Model Prediction':
    
    sc = SparkContext.getOrCreate()
    spark=SparkSession(sc)

    list_Model=["DecisionTree","ExtraTreeClassifier"]
    select_model_mpredict = st.sidebar.selectbox(
        'Select the Model',
        [i for i in list_Model]  
    )

    
    model_dct = joblib.load("Data/Project_1_DecisionTree_sklearn_4.joblib")
    model_ext = joblib.load("Data/Project_1_Extra_sklearn_4.joblib")
    

    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            #data = pd.read_csv(uploaded_file_1)

            check_type=str(uploaded_file_1.type)

            if check_type=="text/csv":
                data = pd.read_csv(uploaded_file_1,sep=";",
                                   names=["comment"], 
                                   header=None, 
                                   engine="python")
                #data.rename(columns={ data.columns[0]: "comment"}, inplace = True)
            elif check_type=="text/plain":
                #data = pd.read_csv(uploaded_file_1, delim_whitespace=",", header = None, names = ['comment'])
                data = pd.read_csv(uploaded_file_1, sep=";", header = None, names = ['comment'],engine="python")
            
            original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Convert Comments to DataFrame</b></p>'
            st.markdown(original_title_duplicate,unsafe_allow_html=True)

            st.dataframe(data)

            data=xu_li_text_dataframe(data)

            if select_model_mpredict=="DecisionTree":
                result = model_dct.predict(data["pre_comment"])
            else:
                result = model_ext.predict(data["pre_comment"])
            
            data["class"]=result
            original_title_duplicate = '<p style="font-family:Garamond, serif; color:Blue; font-size: 30px;"><b>Results</b></p>'
            st.markdown(original_title_duplicate,unsafe_allow_html=True)

            gd_data_pre= GridOptionsBuilder.from_dataframe(data)
            gd_data_pre.configure_pagination(enabled=True)
            gd_data_pre.configure_default_column(editable=True, groupable=True,enableValue=True,enableRowGroup=True)
            gd_data_pre.configure_side_bar()

            sel_mode = 'multiple'
            gd_data_pre.configure_selection(selection_mode=sel_mode, use_checkbox=False)
            gridoptions_reviews = gd_data_pre.build()
            grid_table_reviews = AgGrid(data, gridOptions=gridoptions_reviews,
                                enable_enterprise_modules=True,
                                update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED| GridUpdateMode.MODEL_CHANGED,
                                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                                header_checkbox_selection_filtered_only=False,
                                allow_unsafe_jscode=True,
                                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                                fit_columns_on_grid_load=True)
            
            #st.dataframe(data)

            

            original_title_like = '<p style="font-family:Garamond, serif; color:Black; font-size: 25px;"><b>The predicted value is Like &#128578;</b></p>'
            original_title_dislike = '<p style="font-family:Garamond, serif; color:Black; font-size: 25px;"><b>The predicted value is Dislike &#128577;</b></p>'
            original_title_neutral = '<p style="font-family:Garamond, serif; color:blue; font-size: 25px;"><b>The predicted value is Neutral &#128528;</b></p>'
            
                
            # if result=="Like":
            #     #st.success(f'&#128077 The predicted value is Like')
            #     st.markdown(original_title_like,unsafe_allow_html=True)
            # elif result=="Not_Like":
            #     #st.success(f'The predicted value is DisLike')
            #     st.markdown(original_title_dislike,unsafe_allow_html=True)
            # else:
            #     #st.success(f'The predicted value is Neutral')
            #     st.markdown(original_title_neutral,unsafe_allow_html=True)



    if type=="Input":



        html_str_select_model = f"""
                <style>
                p.a {{
                font: bold 30px Garamond, serif;
                color:blue
                }}
                </style>
                <p class="a"><b>Here is the prediction result with "{select_model_mpredict}" Model</b></p>
                """
        st.markdown(html_str_select_model,unsafe_allow_html=True)
       
        sel_col, disp_col=st.columns(2)

            
        str_1=sel_col.text_input("Input your reviews")


        primaryColor = "#3795BD"

        s = f"""
        <style>
        div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; }}
        <style>
        """
        st.markdown(s, unsafe_allow_html=True)
                        
            
        if st.button('Predict'):
            
            text=[xu_li_text(str_1)]
            
            st.write(xu_li_text(str_1))
            
            text=pd.DataFrame({'pre_comment':text})
            
            if select_model_mpredict=="DecisionTree":
                result = model_dct.predict(text["pre_comment"])
            else:
                result = model_ext.predict(text["pre_comment"])

            original_title_like = '<p style="font-family:Garamond, serif; color:Black; font-size: 25px;"><b>The predicted value is Like &#128578;</b></p>'
            original_title_dislike = '<p style="font-family:Garamond, serif; color:Black; font-size: 25px;"><b>The predicted value is Dislike &#128577;</b></p>'
            original_title_neutral = '<p style="font-family:Garamond, serif; color:blue; font-size: 25px;"><b>The predicted value is Neutral &#128528;</b></p>'
            
                
            if result=="Like":
                #st.success(f'&#128077 The predicted value is Like')
                st.markdown(original_title_like,unsafe_allow_html=True)
            elif result=="Not_Like":
                #st.success(f'The predicted value is DisLike')
                st.markdown(original_title_dislike,unsafe_allow_html=True)
            else:
                #st.success(f'The predicted value is Neutral')
                st.markdown(original_title_neutral,unsafe_allow_html=True)

    
    
