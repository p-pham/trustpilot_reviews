#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", -1)


# In[ ]:


cwd = os.getcwd()
parent_path = os.path.abspath(os.path.join(cwd, os.pardir))


# In[111]:


# trustpilot_fname = 'trustpilot_octopus_20211210.csv'
trustpilot_fname = 'trustpilot_bg_20211209.csv'
trustpilot_output_path = '{}/output/{}'.format(parent_path,trustpilot_fname)
trustpilot_result = pd.read_csv(trustpilot_output_path)
trustpilot_result['emo'] = trustpilot_result['emo'].replace(r'<pad>|</s>','',regex=True)
trustpilot_result['emo'] = [item.strip() for item in trustpilot_result['emo']]
trustpilot_result['keywords'] = trustpilot_result['keywords'].str.replace('[','').str.replace(']','')
trustpilot_result['keythemes'] = trustpilot_result['keythemes'].str.replace('[','').str.replace(']','')
trustpilot_result[['review_rating','title','text','emo','keythemes','keywords']].head()


# In[ ]:


def key_extract(df,cref,cname,new_cname):
    cref.append(cname)
    col_ls = list(set(cref))
    key_df = df.loc[:,col_ls]
    key_df = key_df[key_df[cname]!=''] #remove empty one
    key_df[new_cname] = [item.split(',') for item in key_df[cname]]
    keyword_df = key_df.explode(new_cname).reset_index(drop=True)
    keyword_df[new_cname] = [item.strip() for item in keyword_df[new_cname]] #trim white space
    return keyword_df


# In[70]:


cref = ['text','review_rating','emo']
new_cname = ['extracted_keys','extracted_themes']
key_cname = ['keywords','keythemes']


# In[ ]:


# col_ls = list(set(cref+key_cname))
# key_df = trustpilot_result.loc[:,col_ls]
# key_df[new_cname] = key_df[key_cname].apply(lambda x: [item.split(',') for item in x], axis=1, result_type="expand") #[item.split(',') for item in key_df[key_cname]]
# keywords_df = key_df.explode(new_cname[0]).reset_index(drop=True)
# keythemes_df = key_df.explode(new_cname[1]).reset_index(drop=True)
# keywords_df.head()


# In[72]:


sumarise_keywords = key_extract(trustpilot_result,cref,'keywords','extracted_keys')
sumarise_keywords = key_extract(trustpilot_result,cref,'keywords','extracted_keys')
sumarise_keywords=sumarise_keywords.drop('keywords',1)
sumarise_keywords.head()


# In[73]:


sumarise_themes = key_extract(trustpilot_result,cref,'keythemes','extracted_themes')
sumarise_themes = key_extract(trustpilot_result,cref,'keythemes','extracted_themes')
sumarise_themes=sumarise_themes.drop('keythemes',1)
sumarise_themes.head()


# In[74]:


rate_col = 'review_rating'
rating_summarise = trustpilot_result.groupby(rate_col).size().reset_index(name='counts')
rating_summarise['pct'] = rating_summarise.apply(lambda x: x.counts/len(trustpilot_result.index), axis = 1)
rating_summarise


# In[112]:


colors = sns.color_palette('pastel')[0:5]
plt.pie(rating_summarise['counts'], labels = rating_summarise['review_rating'], colors = colors, autopct='%.0f%%')


# In[180]:


groupby_cols = ['review_rating','extracted_themes']
keywords_count = sumarise_themes.groupby(groupby_cols).size().reset_index(name='counts')
keywords_count.sort_values(by=['review_rating','counts'], ascending=[True,False])['extracted_themes'].head(5)
kw_ls = []
for i in range(1, 6):
    kwi = keywords_count.loc[keywords_count['review_rating'] == i].sort_values(by=['review_rating','counts'], ascending=[True,False])['extracted_themes'].head(10)
    kw_ls.append(kwi)
print(kw_ls)    


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
# Apply the default theme
sns.set_theme()
colors = {'anger': "r", 'fear': "orange", 'joy': "g", 'sadness': "b", 'love': "m", 'surprise': "y"}


# In[183]:


df1 = sumarise_themes.loc[sumarise_themes['review_rating'] == 1 & sumarise_themes['extracted_themes'].isin(kw_ls[0])]
# count plot along y axis
sns.countplot(y ='extracted_themes', data = df1).set_title('1 star rating')
 
# Show the plot
plt.show()


# In[184]:


df5 = sumarise_themes.loc[sumarise_themes['review_rating'] == 1 & sumarise_themes['extracted_themes'].isin(kw_ls[4])]
# count plot along y axis
g = sns.countplot(y ='extracted_themes', data = df5).set_title('5 star rating')

# Show the plot
plt.show()


# In[185]:


key_plot = "'switch provider'"
df_contact_issues = sumarise_themes.loc[sumarise_themes['extracted_themes'] == key_plot]
# count plot along y axis
g = sns.countplot(y ='review_rating',hue = 'emo', data = df_contact_issues, palette = colors).set_title(key_plot)

# Show the plot
plt.show()


# In[82]:


with pd.option_context('display.max_rows', None): 
    display(keywords_count.sort_values([rate_col,'extracted_keys','counts'], ascending=[True,True,False]))
# print(sumarise_count.sort_values([rate_col,'counts'], ascending=[True,True]))


# In[ ]:




