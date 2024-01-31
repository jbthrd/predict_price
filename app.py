from flask import Flask, request, jsonify
import pandas as pd
import os
import pandas as pd
import numpy as np
import deepl
import re
from google.cloud import storage
import joblib
import os

# Natural Language Processing Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer, wordnet
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Machine Learning and Data Preprocessing Libraries
import scipy.sparse
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_log_error

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.python.keras.metrics import mean_squared_logarithmic_error
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import optimizers


app = Flask(__name__)


#Storing gcloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './retroc-service-account.json'


#First step : Map the object to DataFrame format
def object_to_df(obj):
    data = {
        'name': [obj.get('name', '')],
        'description': [obj.get('description', '')],
        'etat': [obj.get('etat', '')],
        'category': [obj.get('category', '')],
        'subcategoryName': [obj.get('subcategoryName', '')],
        'item_condition': [obj.get('etat', '')],
        'brand_name': ''  # Placeholder for brand_name
    }
    return pd.DataFrame(data)



#Second step : Translate object elements
auth_key = "96434628-8786-869f-6249-8a0073d763cd"  # Retroc pro key
translator = deepl.Translator(auth_key)

def translate_to_english(sentence): #ugly but efficient
    if sentence != '' :
      translation = translator.translate_text(sentence, source_lang="FR", target_lang="EN-US")
      return translation.text
    else :
      return 5/0 #to create an error
    

#Third step : used to translate the categories
def translate_column(df, column_name):
    unique_values = df[column_name].dropna().unique() #no pb with empty strings
    translated_values = {'' : ''}

    for value in unique_values:
        if value != '' :
          translated = translator.translate_text(value, source_lang="FR", target_lang="EN-US").text
          translated_values[value] = translated if translated else value

    new_colname = column_name + "_translated"
    df[new_colname] = df[column_name].map(translated_values) #I prefer to define another column

    return df, translated_values



# 4th functions : Defining preprocessing functions

#REGEX Expressions
def decontracted(phrase):
    '''
    Description -> Replaces the short form words to their decontracted form such as won't to will not,
                   this is done to make the text data uniform using regex commands.
    '''
    phrase = re.sub(r"aren\'t", "are not", phrase)
    phrase = re.sub(r"didn\'t", "did not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"couldn\'t", "could not", phrase)
    phrase = re.sub(r"won\'t", "would not", phrase)
    phrase = re.sub(r"wouldn\'t", "would not", phrase)
    phrase = re.sub(r"haven\'t", "have not", phrase)
    phrase = re.sub(r"shouldn\'t", "should not", phrase)
    phrase = re.sub(r"doesn\'t", "does not", phrase)
    phrase = re.sub(r"don\'t", "do not", phrase)
    phrase = re.sub(r"didn\'t", "did not", phrase)
    phrase = re.sub(r"mustn\'t", "must not", phrase)
    phrase = re.sub(r"needn\'t", "need not", phrase)

    return phrase



def perform_lemma(sent,all_stopwords):
    '''
    Description -> Applying wordnet lemmatizer on the input sentence and returning the nearest base word/sentence.
    '''
    sent_list = sent.split()
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in sent_list if word not in all_stopwords]
    sent = " ".join(text)
    return sent



def process_text_data(sent, regex_special_chars, regex_decimal_digits, regex_white_space, all_stopwords):
    '''
    Description -> Calling the demojify and lemmatization functions one by one with data as the input
                  and returning the final preprocessed data.
    '''

    #Removing new line, carriage return, double quotes
    sent = sent.replace('\\r', ' ')
    #sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')

    #removing all special charecter except mentioned in regex strings
    sent = regex_special_chars.sub(' ', sent)

    #removing all digit mentioned in regex strings
    sent = regex_decimal_digits.sub(' ', sent)

    # removing all white spaces  https://bobbyhadz.com/blog/python-remove-whitespace-regex
    sent = regex_white_space.sub(' ', sent)
    #sent = re.sub(r'^\s+', '', sent, flags=re.MULTILINE)

    #removing end space and converting to lower
    sent = sent.strip().lower()

    # only take the words which are not stop words
    sent = perform_lemma(sent, all_stopwords)
    return sent


# 5th functions : Text encoding
def tfidf_transform(to_tf, vectorizer) :
    """Transforms some data with a vectorizer"""
    tf = vectorizer.transform(to_tf)
    return tf


# 6th functions : Load files from Google cloud Storage
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


# 7th functions : Model architecture
    
## MLP model architecture
def SMLP(frame_shape,i):
  input = Input(shape=frame_shape,sparse=True,dtype='float32')
  output = Dense(i,activation='relu')(input)
  output = BatchNormalization()(output)
  if i==256:
    while i!=64:
      i/=2
      output = Dense(i,activation='relu')(output)
  else:
    while i!=16:
      i/=2
      output = Dense(i,activation='relu')(output)
  output = Dropout(0.2)(output)
  output = Dense(1)(output)
  model = Model(input,output)
  return model

def loss(y_true,y_pred):
    """
    Description -> Accepts input y1 and y2 as the true and predicted values respectively to determine the error
                   loss and returns it
    """
    return np.sqrt(mean_squared_log_error(y_true,y_pred))

def manual_pricing(price):
    if price <= 5:
        return '000-005' #05 for nice alphabetical ordering
    elif price <= 8:  # Changed from 6 <= price <= 8
        return '005-008'
    elif price <= 10:  # Changed from 9 <= price <= 10
        return '008-010'
    elif price <= 16:  # Changed from 11 <= price <= 16
        return '010-016'
    elif price <= 20:  # Changed from 17 <= price <= 20
        return '016-020'
    elif price <= 26:  # Changed from 21 <= price <= 26
        return '020-026'
    elif price <= 36:  # Changed from 27 <= price <= 36
        return '026-036'
    elif price <= 58:  # Changed from 37 <= price <= 58
        return '036-058'
    elif price <= 84:  # Changed from 59 <= price <= 84
        return '058-084'
    elif price <= 170:  # Changed from 85 <= price <= 170
        return '084-170'
    else:
        return '170+'






# Specify your bucket name
bucket_name = 'training_data_price_prediction'













    


def item_unit_pricer(obj):

###First step : Map the object to DataFrame format###
    df = object_to_df(obj)
###First step : Map the object to DataFrame format###
    
    
###Second step : handle translation###
    df['en_name'] = 'no_name'
    df.loc[~df['name'].isnull(), 'en_name'] = df[~df['name'].isnull()]['name'].apply(translate_to_english)
    df.head() #I think it's not a bad idea to keep such commands even if they are not "active".

    df['en_description'] = 'no_description'
    df.loc[~df['description'].isnull(), 'en_description'] = df[~df['description'].isnull()]['description'].apply(translate_to_english)
    df.head()
    print("Finished the translation. Now just translating column names.")

    df['Retroc_category'] = df['category'] # I think this will feel strange in the future (why indicate Retroc ??) but you can always change it and it will simplify my current work
    df['Retroc_subcategory'] = df['subcategoryName']

    df.fillna('', inplace=True) #important for translation
    #Checkpoint
    df.to_csv('df_with_translations.csv')

    
    #Translating the categories
    df, translated_cats_map = translate_column(df, 'Retroc_category')
    df, translated_subcats_map = translate_column(df, 'Retroc_subcategory')
    df.head()
    print("Translated the categories. Now mapping the item condition.")
###Second step : handle translation###


###Third step : Define the mapping ###
    condition_mapping = {
        'Neuf': 1,
        'Comme neuf': 2,
        'Bon état': 3,
        'Bon': 3,
        'État moyen': 4,
        'Passable': 4,
        'Abîmé': 5
    }

    # Apply the mapping to the item_condition column
    df['item_condition_id'] = df['item_condition'].replace(condition_mapping)

    df['item_condition_id'].value_counts()


    print("Dealt with item condition. Now starting the preprocessing.")



    df['brand_name'] = None
### Third step : Define the mapping ###
    

### 4th step : Pre-processing ###
    df.fillna('', inplace=True)
    # Instantiating the category name
    df['category_name'] = df['Retroc_category_translated'] + ' - ' +  df['Retroc_subcategory_translated']


    # Data preprocessing

    df['en_description'] = df['en_description'].str.replace('^no description yet$', '', regex=True)

    print("Dealt with null values")

    ## Text features
    df['name'] = df['en_name'] + " " + df['brand_name'] #brand_name is allways empty for the moment but it's coming !
    df['text'] = df['en_description'] + " " + df['en_name'] + " " + df['category_name']

    print("Built text and name features")


     ###################### Applying our preprocessing functions
    ## Text cleaning

    df['name'] = df['name'].apply(lambda x : decontracted(x))
    df['text'] = df['text'].apply(lambda x : decontracted(x))

    print("Applied the decontraction")

    ## Lemmatization and final textual pre-processing
    regex_special_chars = re.compile('[^A-Za-z0-9.]+')
    regex_decimal_digits = re.compile('(?<!\d)\.(?!\d)')
    regex_white_space = re.compile(r'\s+')

    all_stopwords = set(stopwords.words("english")) - {"no", "nor", "not"} #Negation is super important !


    df['name'] = df['name'].apply(lambda x : process_text_data(x, regex_special_chars, regex_decimal_digits,
                                                                  regex_white_space, all_stopwords))
    df['text'] = df['text'].apply(lambda x : process_text_data(x, regex_special_chars, regex_decimal_digits,
                                                                  regex_white_space, all_stopwords))

    print("Applied final text preprocessing")

### 4th step : Pre-processing ###
    


### 5th step : Test-encoding  ###

    # Download the fitted TFIDF vectorizers 
    download_blob(bucket_name, 'Production/name_basic_vecto.joblib', './name_basic_vecto.joblib')
    download_blob(bucket_name, 'Production/text_basic_vecto.joblib', './text_basic_vecto.joblib')

    #Loading the fitted TFIDFs 
    name_tfidf = joblib.load('./name_basic_vecto.joblib')
    text_tfidf = joblib.load('./text_basic_vecto.joblib')

    ### Applying TFIDF
    print("Loaded big_name and big_tetx. Applying TFIDF. ")

    X_name = tfidf_transform(to_tf=df['name'],
                            vectorizer=name_tfidf)
    X_text = tfidf_transform(to_tf=df['text'],
                            vectorizer=text_tfidf)

    print("Applied TFIDF on name and text feature.")



   ## One hot encoding

    # Download and load the one_hot_encoder
    download_blob(bucket_name, 'Production/one_hot_encoder.joblib', './one_hot_encoder.joblib')
    ohe = joblib.load('./one_hot_encoder.joblib')

    X_item_condition = ohe.transform(np.reshape(df['item_condition_id'].values, (-1, 1))).shape

    print("One-hot encoded item condition.")

    # Stacking
    frame = hstack((X_name,
                    X_text,
                    X_item_condition)).tocsr().astype('float32')


    print("Stacked features into trainframe")
   
    print("Finished the preprocessing. Now loading models' weights.")

### 5th step : Test-encoding  ###
    

### 6th step : Loading the weights and initializing infrastructure  ###
    #Need to cache ??
     #Importing from GCS
    download_blob(bucket_name, 'Production/model1_weights.hdf5', './model1_weights.hdf5')
    download_blob(bucket_name, 'Production/model2_weights.hdf5', './model2_weights.hdf5')

    #Initializing models infrastructures
    model1 = SMLP(frame.shape[1], 256)
    model1.compile(optimizer='adam', loss=['mean_squared_error'], metrics=['mean_squared_error'])

    model2 = SMLP(frame.shape[1], 1024)
    model2.compile(optimizer='adam', loss=['mean_squared_error'], metrics=['mean_squared_error'])

    #Loading
    model1.load_weights('./model1_weights.hdf5')
    model2.load_weights('./model2_weights.hdf5')



    print("Models laoded. Generating prediction.")
### 6th step : Loading the weights  ###
    
### 7th Step : Generating prediction  ###

    y_pred_model1 = model1.predict(frame)
    y_pred_model2 = model2.predict(frame)

    wmin = 0.463
    y_pred_ensemble = wmin * y_pred_model1 + (1-wmin) * y_pred_model2


    # We don't store the model 1 prices and model 2 prices.
    df['price_ensemble'] = np.exp(y_pred_ensemble) - 1

    # Add price class columns to df_test

    df['price_cat_ensemble'] = df['price_ensemble'].apply(manual_pricing)

    print("Here we go.")
    df.head()
### 7th Step : Generating prediction  ###


    return df[0]['price_cat_ensemble']





@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json()
    if not data:
        return 'Invalid or missing JSON object in request', 400
    
    df = item_unit_pricer(data)
    # Convert DataFrame to JSON
    result = df.to_json(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))