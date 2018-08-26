
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text


# In[3]:


len(documents)


# In[4]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)


# In[5]:


import nltk
nltk.download('wordnet')


# In[6]:


print(WordNetLemmatizer().lemmatize('went', pos='v'))


# In[8]:


stemmer = SnowballStemmer('english')
original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
           'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in original_words]
pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})


# In[10]:


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[11]:


doc_sample = documents[documents['index'] == 4310].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[12]:


processed_docs = documents['headline_text'].map(preprocess)


# In[ ]:


processed_docs[:10]


# In[13]:


dictionary = gensim.corpora.Dictionary(processed_docs)


# In[14]:


count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[15]:


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[16]:


bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4310]


# In[17]:


bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                                     dictionary[bow_doc_4310[i][0]], 
                                                     bow_doc_4310[i][1]))


# In[18]:


from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)


# In[19]:


corpus_tfidf = tfidf[bow_corpus]


# In[20]:


from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


# In[22]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)


# In[23]:




for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))



# In[ ]:


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)


# In[ ]:


processed_docs[4310]


# In[ ]:


for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# In[ ]:


for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# In[ ]:


unseen_document = '05/10/2015 AT 04:36 PM HMLO000000  INSURANCE COMPANY ATLANTA-CLAIM DEPARTMENT- PI FOR SUPPLEMENT INSPECT CALL 888-299-3456 PROMPT 2 1001 WINDWARD CONCOURSE, SUITE 10 ALPHARETTA, GA 30005-2023 (800) 238-6200  ESTIMATE OF RECORD  WRITTEN BY: AUTO DAMAGE APPRAISER 05/10/2015 04:36 PM ADJUSTER: CLAIMS ADJUSTER #1 (678) 317-7400  INSURED: JOHN WILKES BOOTH CLAIM #HMLO000000 OWNER: LEE HARVEY OSWALD POLICY # 012345687 ADDRESS: 1000 NEW HOPE RD DATE OF LOSS: 05/09/2015 AT 09:40 AM ATLANTA, GA 30331 TYPE OF LOSS: LIABILITY CELLULAR: (404) 770-6789 POINT OF IMPACT: 6. REAR INSPECT STARBUCKS COFFEE SHOP DAY: (404) 888-7777 LOCATION: CASCADE ROAD OTHER  ATLANTA, GA 30331-0000  REPAIR 3 DAYS TO REPAIR FACILITY: COLLISION SHOPPE LICENSE #  2007 TOYO FJ-CRUISER 4X2 6-4.0L-FI 4D UTV SILVER INT:GREY VIN: JTEZU11F6700122345 LIC: AVI6789 GA PROD DATE: 01/2007 ODOMETER:142492  CONDITION: GOOD  AIR CONDITIONING REAR DEFOGGER TILT WHEEL  INTERMITTENT WIPERS PARKING SENSORS DUAL MIRRORS  CONSOLE/ STORAGE LUGGAGE/ROOF RACK CLEAR COAT PAINT  METALLIC PAINT POWER STEERING POWER BRAKES  POWER WINDOWS POWER LOCKS AM RADIO  FM RADIO STEREO SEARCH/SEEK  CD PLAYER ANTI-LOCK BRAKES (4) DRIVER AIR BAG  PASSENGER AIR BAG 4 WHEEL DISC BRAKES TRACTION CONTROL  STABILITY CONTROL CLOTH SEATS BUCKET SEATS  RECLINE/LOUNGE SEATS AUTOMATIC TRANSMISSION ALUMINUM/ALLOY WHEELS NO. OP. DESCRIPTION QTY EXT. PRICE LABOR PAINT  1 REAR BUMPER  O/H REAR BUMPER 2.4  WWW . DIMINISHEDVALUEOFGEORGIA.COM    N 3 REPL LOWER COVER TO 1/07 1 89.96 INCL. 0.6 4 REPL PREP UNPRIMED BUMPER LOWER 1 0.2 COVER 5** REPL RECOND BUMPER COVER W/REVERSE 1 212.00 INCL. SENSOR 6 ADD FOR REVERSE SENS M 0.3 N 7 REPL LT CORNER PAD TO 1/07 1 138.32 INCL. 0.6 8 ADD FOR CLEAR COAT 0.1 9 REPL PREP UNPRIMED BUMPER CORNER 1 0.2 PAD 1 05/10/2015 AT 04:36 PM HMLO00000 ESTIMATE OF RECORD 2007 TOYO FJ-CRUISER 4X2 6-4.0L-FI 4D UTV SILVER INT:GREY NO. OP. DESCRIPTION QTY EXT. PRICE LABOR PAINT 10 REAR LAMPS 11 REPL LT REFLECTOR 1 28.90 INCL. 12 R&I RT REFLECTOR INCL. 13 FENDER N 14* REPL LT FILLER TO BUMPER SILVER 1 34.51 0.1 0.0* 15 ELECTRICAL N 16 REPL RT REVERSE SENSOR 1 182.02 0.1 1 REPL LT REVERSE SENSOR 1 182.02 0.1 18 FRONT BUMPER 19 O/H FRONT BUMPER 2.0 20** REPL RECOND BUMPER COVER 1 185.00 INCL. 2.4 21 ADD FOR CLEAR COAT 1.0 N 22* REPL LT CORNER PAD 1 104.84 INCL. 0.0* N 23* RPR VALANCE PANEL TO 1/07 0.5* 0.8 24 OVERLAP MINOR PANEL -0.2 25 REPL UPPER RETAINER 1 55.80 0.2 26 REPL LOWER GRILLE 1 92.98 INCL. Zt OTHER CHARGES 284 HAZARDOUS WASTE 1 3.00 SUBTOTALS ==> 1309.35 5.7 5.7 LINE 3. : PARTS: COMPONENT COMES UNPRIMED FROM OEM. PREPARATION IS REQUIRED. SEE ADD IF  REQUIRED OPERATION. LINE 7 : PART NOT RECOVERED FROM ACCIDENT SCENE. PARTS: COMPONENT COMES UNPRIMED FROM OEM. PREPARATION IS REQUIRED.  WWW . DIMINISHEDVALUEOFGEORGIA.COM    SEE ADD IF  REQUIRED OPERATION. LINE 14 : NOT RECOVERED FROM ACCIDENT SCENE, PART SUPPLIED IN COLOR. LINE 16 : OUTER RING DAMAGED, NOT SHOWN SERVICED SEPARATELY. LINE 22 : PER DATABASE PART CAN BE ORDERED IN COLOR. LINE 23 : SCUFFED & SCRATCHED.  ESTIMATE NOTES:  *REVIEWED DAMAGES, REPAIR PROCESS & UPD W/MR. JONES, MET BRIEFLY WITH OWNER AT COFFEE SHOP, COPY OF APPRAISAL EMAILED TO HIM TO JONESROYCE@COMCAST.NET.  *LKQ AVAILABLE: RPS, APU. *INSP MAY BE NEEDED AFTER TEAR DOWN.  *REVIEW RENTAL POLICY WITH MR. JONES, OWNER HAS CHOSEN ACE BODY SHOP FOR THE REPAIRS.  05/10/2015 AT 04:36 PM HMLO000000  ESTIMATE OF RECORD 2007 TOYO FJ-CRUISER 4X2 6-4.0L-FI 4D UTV SILVER INT:GREY  PRIOR DAMAGE NOTES:  RT FRONT FLARE SCRAPED.  RT DOOR DENTED AT REAR EDGE. DOOR DINGS ON LT DOOR.  VEHICLE DIRTY WHEN INSPECTED.  RT SIDE REAR BUMPER END SCRAPED.  PARTS 1306.35 BODY LABOR 5.7 HRS @$ 40.00/HR 228.00 PAINT LABOR 5.7 HRS @$ 40.00/HR 228.00 PAINT SUPPLIES 5.7 HRS @$ 28.00/HR 159.60 OTHER CHARGES 3.00 SUBTOTAL $ 1924.95 SALES TAX $ 1465.95 @ 7.0000% 102.62 TOTAL COST OF REPAIRS $ 2027.57 TOTAL ADJUSTMENTS $ 0.00  NET COST OF REPAIRS $ 2027.57  WWW . DIMINISHEDVALUEOFGEORGIA.COM'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

