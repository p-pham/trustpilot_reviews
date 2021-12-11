#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", -1)
import os
import json
import time

import pke
# import nltk
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
stoplist.append(['British Gas','british gas'])


# In[2]:


# from keybert import KeyBERT
# kw_model = KeyBERT()
# def bert_kw(t,keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=3):
#     keywords = kw_model.extract_keywords(t,
#     keyphrase_ngram_range=keyphrase_ngram_range, 
#     stop_words=stop_words, 
#     use_maxsum=use_maxsum, 
#     nr_candidates=nr_candidates, 
#     top_n=top_n)
#     return keywords


# In[3]:


# bert_kw('British Gas were very accommodating.and first class service.')


# In[4]:


from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
t5_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")


# In[5]:


def db_condition(x,x_thres): return x >= x_thres
# def yake_condition(x,x_thres): return x <= x_thres


# ### Yake keywork extractor

# In[6]:


def yake_ke(t,yake_threshold,n,stoplist):

    # 1. create a YAKE extractor.
    extractor = pke.unsupervised.YAKE()

    # 2. load the content of the document.
    extractor.load_document(t)

    # 3. select {1-3}-grams not containing punctuation marks and not
    #    beginning/ending with a stopword as candidates.
    # stoplist = stopwords.words('english')
    extractor.candidate_selection(n=3, stoplist=stoplist)

    # 4. weight the candidates using YAKE weighting scheme, a window (in
    #    words) for computing left/right contexts can be specified.
    window = 2
    use_stems = False # use stems instead of words for weighting
    extractor.candidate_weighting(window=window,
                                  stoplist=stoplist,
                                  use_stems=use_stems)

    # 5. get the 10-highest scored candidates as keyphrases.
    #    redundant keyphrases are removed from the output using levenshtein
    #    distance and a threshold.
    keywords_ls = extractor.get_n_best(n=n, threshold=yake_threshold)
    yake_keywords = [k[0] for k in keywords_ls]
    yake_scores = [k[1] for k in keywords_ls]
    idx_ls = [idx for idx, element in enumerate(yake_scores)]
    yake_keywords = [yake_keywords[i] for i in idx_ls]
    yake_scores = [yake_scores[i] for i in idx_ls]
    yake_ls = pd.Series([yake_keywords,yake_scores])
    return yake_ls


# In[7]:


# yake_ke('British Gas were very accommodating.and first class service.', 0.5,10,stoplist)


# In[8]:


# import yake
# def yake_ke(t,lang="en", ngram=3, ddlim=0.9, ddfn='seqm', wz=1, num_kw=3, features=None, yake_score_thres = 0.15):
#     """ 
#     Using Yake to extract key words
#     """
#     custom_kwextractor = yake.KeywordExtractor(lan = lang, n = ngram, dedupLim = ddlim, dedupFunc = ddfn, windowsSize = wz, top = num_kw, features = features)
#     keywords_ls = custom_kwextractor.extract_keywords(t)
# #     keywords_df = pd.DataFrame(keywords_ls, columns = ['yake_kw','yake_scores'])
#     yake_keywords = [k[0] for k in keywords_ls]
#     yake_scores = [k[1] for k in keywords_ls]
#     idx_ls = [idx for idx, element in enumerate(yake_scores) if yake_condition(element,yake_score_thres)]
#     yake_keywords = [yake_keywords[i] for i in idx_ls]
#     yake_scores = [yake_scores[i] for i in idx_ls]
#     yake_ls = pd.Series([yake_keywords,yake_scores])
#     return yake_ls


# ### DistilBart-MNLI

# In[9]:


def distillBart_ke(t,cand_lables,num_kw=3,db_score_thres = 0.095):
    db_output = classifier(t,cand_lables)
    db_keywords = db_output.get('labels')[:num_kw]
    db_scores = db_output.get('scores')[:num_kw]
    idx_ls = [idx for idx, element in enumerate(db_scores) if db_condition(element,db_score_thres)]
    db_keywords = [db_keywords[i] for i in idx_ls]
    db_scores = [db_scores[i] for i in idx_ls]
    db_ls = pd.Series([db_keywords,db_scores])
    return db_ls


# **T5 emotion**

# In[10]:


### t5-base-finetuned-emotion
def get_emotion(text):
    input_ids = t5_tokenizer.encode(text + '</s>', return_tensors='pt')

    output = t5_model.generate(input_ids=input_ids, max_length=512)

    dec = [t5_tokenizer.decode(ids) for ids in output]
    label = dec[0]
#   
    return label


# In[11]:


# my_labels = ['contact issues', 
#                     'payment issues', 
#                     'billing issues',
#                     'app issues',
#                     'account issues',
#                     'smart meter', 'meter issues',
#                     'poor customer service', 'good customer service',
# #                     'vulnerable customer', 
# #                     'expensive', 'cheap',
#                     'over charged', 'reasonable price',
#                     'price increase',
#                     'no show','reschedule',
#                     'trustworthy'
#                    ]
# promotor_lables=[
# 'good communication',
# 'easy account management',
# # 'fulfilling needs',
# 'good tariff',
# 'successful switch',
# 'efficient',
# 'quick',
# 'easy',
# 'reassuring',
# # 'professional',
# 'simple',
# 'helpful'
# ]
# detractor_lables = [
# 'dishonest',
# 'complicated',
# 'confusing',
# # 'useless',
# # 'mistakes',
# 'untrustworthy',
# 'uninterested',
# 'difficult to switch',
# # 'smart metering issues',
# 'difficult online account management access',
# 'homecare issues',
# 'poor communication',
# # 'issues with billing',
# # 'homemove',
# 'direct debit set-up issues',
# 'direct debit amendment issues',
# 'failure complaints handling',
# 'poor website experience',
# 'meter read frequency ',
# 'poor migration experience'
# ]

# candidate_labels = list(set(my_labels + promotor_lables + detractor_lables))
# len(candidate_labels)


# In[12]:


# cand_keyemo = [
#     'trustworthy',
#     'efficient',
#     'quick',
#     'easy',
#     'reassuring',
#     'professional',
#     'simple',
#     'helpful',
#     'happy',
#     'dishonest',
#     'complicated',
#     'confusing',
#     'useless',
#     'mistakes',
#     'untrustworthy',
#     'uninterested',
#     'not impressed',
#     'frustrated',
#     'angry'
# ]
cand_keythemes = [
    'metering','smart meter',
    'communication',
    'contact',
    'billing',
    'payment',
    'price','tariff',
    'switch supplier', 'switch provider',
    'home move',
    'homecare',
    'boiler',
    'heating',
    'vulnerable customer',
    'engineer',
    'account management',
    'direct debit',
    'engineer visit',
    'reschedule',
    'annual service',
    'hive'
]
cand_keywords = [
    'contact issues', 'poor communication','good communication',
    'payment issues', 'direct debit set-up issues','direct debit amendment issues',
    'billing issues',
    'app issues','easy to use app',
    'good website experience','poor website experience',
    'account issues','difficult online account management access','easy account management',
    'smart meter working fine','smart meter issues', 'meter issues', 'meter read frequency','incorrect read',
    'good tariff','reasonable price', 
    'over charged', 'price increased',
    'no show','reschedule',
    'successful switch supplier','difficult to switch supplier','poor migration',
    'good customer experience','poor customer experience',
    'homecare issues',
    'home move issues','home move success',
    # 'fulfilling needs',
    'failure complaints handling',
    'faulty boiler','boiler issues',
    'heating issues','no heating','no hot water',
    'gas leak', 'water leak'
]


# In[13]:


num_text = 1000
num_key = 3
yake_threshold = 0.6
db_score_threshold = 0.1


# In[14]:


cwd = os.getcwd()
parent_path = os.path.abspath(os.path.join(cwd, os.pardir))


# ### Test on trustpilot reviews

# In[15]:


trust_cache_path = '{}/cache/trust_cache_file_bg_20211209.json'.format(parent_path)
with open(trust_cache_path) as f:
    trustpilot_reviews_dts = json.load(f)
review_values = list(trustpilot_reviews_dts.values())
treview_df = pd.DataFrame.from_dict(review_values)
test_df = treview_df[:num_text]
# test_df[['ykeywords','ykw_scores']] = test_df.loc[:,'text'].apply(yake_ke, num_kw = num_key,yake_score_thres = yake_score_threshold)
# test_df['bert_keyword'] = test_df.loc[:,'text'].apply()
# test_df[['keyemos','keyemos_scores']] = test_df.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keyemo, num_kw = num_key, db_score_thres = db_score_threshold)
test_df[['ykeywords','ykw_scores']] = test_df.loc[:,'text'].apply(yake_ke,yake_threshold = yake_threshold,n = num_key, stoplist= stoplist)
test_df['emo'] = test_df.loc[:,'text'].apply(get_emotion)
test_df[['keythemes','keythemes_scores']] = test_df.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keythemes, num_kw = num_key, db_score_thres = db_score_threshold)
test_df[['keywords','keywords_scores']] = test_df.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keywords, num_kw = num_key, db_score_thres = db_score_threshold)
# test_df


# ### Test on KPMG data copied from dashboard

# In[16]:


# kpmg_path = '{}/kpmg_samples/BG_Evolve_NPSCommentsExport.txt'.format(parent_path)
# kpmg_data = pd.read_csv(kpmg_path, delimiter = "\t", encoding='cp1252')
# kpmg_data = kpmg_data[kpmg_data['NPS_Verbatim'].notna()]
# # num_text = len(kpmg_data.index)
# kpmg_test = kpmg_data[:num_text]
# kpmg_test[['ykeywords','ykw_scores']] = kpmg_test.loc[:,'text'].apply(yake_ke, num_kw = num_key,yake_score_thres = yake_score_threshold)
# kpmg_test[['keyemos','keyemos_scores']] = kpmg_test.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keyemo, num_kw = num_key, db_score_thres = db_score_threshold)
# kpmg_test[['keythemes','keythemes_scores']] = kpmg_test.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keythemes, num_kw = num_key, db_score_thres = db_score_threshold)
# kpmg_test[['keywords','keywords_scores']] = kpmg_test.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keywords, num_kw = num_key, db_score_thres = db_score_threshold)
# # kpmg_test
# num_text


# ### Save output

# In[17]:


version = time.strftime("%Y%m%d")
trust_fname = '{}/output/trustpilot_bg_{}.csv'.format(parent_path,version)
# kpmg_fname = '{}/output/kpmg_test_{}.csv'.format(parent_path,version)


# In[18]:


test_df.to_csv(trust_fname)
# kpmg_test.to_csv(kpmg_fname)


# In[19]:


trust_cache_path_o = '{}/cache/trust_cache_file_octopus_20211209.json'.format(parent_path)
with open(trust_cache_path_o) as f:
    trustpilot_reviews_dts_o = json.load(f)
review_values_o = list(trustpilot_reviews_dts_o.values())
treview_df_o = pd.DataFrame.from_dict(review_values_o)
test_df_o = treview_df_o[:num_text]
# test_df[['ykeywords','ykw_scores']] = test_df.loc[:,'text'].apply(yake_ke, num_kw = num_key,yake_score_thres = yake_score_threshold)
# test_df['bert_keyword'] = test_df.loc[:,'text'].apply()
# test_df[['keyemos','keyemos_scores']] = test_df.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keyemo, num_kw = num_key, db_score_thres = db_score_threshold)
test_df_o[['ykeywords','ykw_scores']] = test_df_o.loc[:,'text'].apply(yake_ke,yake_threshold = yake_threshold,n = num_key, stoplist= stoplist)
test_df_o['emo'] = test_df_o.loc[:,'text'].apply(get_emotion)
test_df_o[['keythemes','keythemes_scores']] = test_df_o.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keythemes, num_kw = num_key, db_score_thres = db_score_threshold)
test_df_o[['keywords','keywords_scores']] = test_df_o.loc[:,'text'].apply(distillBart_ke, cand_lables = cand_keywords, num_kw = num_key, db_score_thres = db_score_threshold)
# test_df


# In[20]:


version = time.strftime("%Y%m%d")
o_trust_fname = '{}/output/trustpilot_octopus_{}.csv'.format(parent_path,version)
test_df_o.to_csv(o_trust_fname)


# In[22]:


test_df.head()


# In[21]:


test_df_o.head()


# In[ ]:




