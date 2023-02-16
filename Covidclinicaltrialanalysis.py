#!/usr/bin/env python
# coding: utf-8

# ## Introduction 
# 
# This analysis focuses on the number of COVID-19 clinical trials that have been registered on https://clinicaltrials.gov/. In order to effectively evaluate the data, it is useful to be familiar with what the ClinicalTrials.gov is and some common clinical trial terminology.
# 
# What is ClinicalTrials.gov?
# 
# It is a database developed by U.S. National Library of Medicine, hosting information of private and public funded clinical trials. 
# 
# `Clinical Trial Terminology from the clinicaltrials.gov`
# 
# | Term       | Definition                                                 |
# |------------|------------------------------------------------------------|
# | Clinical trial | A research study that involves human participants to evaluate a medical treatment or procedure. |
# | Placebo      | A harmless substance that looks and tastes like a medication, but does not contain any active ingredients. |
# | Double-blind | A type of study design in which neither the participants nor the researchers know which participants are receiving the treatment or the placebo. |
# | Phase 2 | A phase of clinical research where preliminary data on whether a drug works in people who have a certain condiction/disease.In this phase, the participants receiving the intervention are compared to those receiving a placebo to determine the drug's effectiveness.|
# | Phase 3| A phase where the drug's safety is studied in different populations and different dosages and the drug in combination with other drugs. The number of participants in Phase 3 is larger than Phase 2.| 
# | Phase 4 | This phase of research occurs after FDA approval on a drug.They include postmarket requirement and commitment studies that are required of or agreed to by the study sponsor.These trials gather additional information about a drug's safety, efficacy, or optimal use.|   
#     
# ** Please note that the source of the datasets is Kaggle- https://www.kaggle.com/code/parulpandey/eda-on-covid-19-clinical-trials/data. 
# 
# 
# 
# 

# `Analysis questions`: 
# 
# * How many clinical trials were conducted in each phase (2-4) where the condition was related to COVID?
# * What phase completed the most covid clinical trials? 
# * Who are the top five sponsors for clinical trials on COVID related conditions?
# 

# Per CDC, the COVID-19 outbreak started in USA on January 18, 2019. 
# And the first clinical trial conducted for COVID-19 was on March 17, 2020 at The University of Minnesota. 
# 
# Source- https://www.cdc.gov/museum/timeline/covid19.html#:~:text=CDC%20reports%20the%20first%20laboratory,respond%20to%20the%20emerging%20outbreak.

# #### Importing Necessary Libaries

# In[1]:


get_ipython().system('pip install ipython-sql')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading in the data with Pandas

# In[3]:


# Reading in the data
cct = pd.read_csv('COVID clinical trials.csv')


# In[4]:


cct.head()


# #### Assessment of data quality issues 

# In[5]:


cct.info()


# In[6]:


cct.shape


# In[7]:


# columns with null values 
cct_null = cct.columns[cct.isnull().any()]
cct_null


# In[8]:


#number of columns with null values
len(cct_null)


# In[9]:


cct.info()


# In[10]:


# Duplicated values in the cct dataframe if any
cct.duplicated().sum()


# In[11]:


# Visual Assessment 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
cct.sample(20)


# ## `Data Quality Issues`
# 
# CCT DataFrame:
# 
# 1. Completeness- Missing values in the following columns, the columns that are unecessary for this analysis are highlighted and can be dropped from cct df. The missing information in columns that are unhighlighted cannot be recovered therefore the rows with missing values will deleted in the copy of the dataframe. 
# 
# `Acronym`
# 
# Interventions
# 
# `Outcome Measures`
# 
# `Gender`
# 
# Phases
# 
# Enrollment
# 
# `Study Designs`
# 
# `Other IDs`
# 
# Start Date
# 
# Primary Completion Date
# 
# Completion Date 
# 
# Results First Posted
# 
# Locations
# 
# `Study Documents`
# 
# 2. Validity: The datatypes of the following columns need to be datetime: Start Date, Completion Date.

# ## `Data Cleaning`

# This section will describe the process of resolving data quality issues by defining the issue, outlining the cleaning method, coding the cleaning process, and testing to confirm that the issue has been resolved."

# In[12]:


cct_copy = cct.copy()


# In[13]:


cct_copy.info()


# #### It is best practice to address the completeness issue of a dataframe. 

# `Define`
# 
# Dropping the following columns by capturing all series in a list and then passing them thru a df.drop(series, axis=0)function.
# 
# Acronym
# 
# Outcome Measures
# 
# Gender
# 
# Study Designs
# 
# Other IDs
# 
# Study Documents

# In[14]:


cct_copy.head()


# `Code`

# In[15]:


cct_copy.drop(columns=['Acronym', 'Outcome Measures', 'Gender', 'Study Designs', 'Other IDs', 'Study Documents' ], inplace=True)


# `Test`

# In[16]:


cct_copy.info()


# `Define`: Change the datetype of the `Start Date and Completion Date` to datetime by using pd.to_datetime. 

# In[17]:


cct_copy[['Start Date', 'Completion Date']] = cct_copy[['Start Date', 'Completion Date']].apply(pd.to_datetime)


# In[18]:


cct_copy.info()


# ### How many covid related clinical trials were completed in each of the phases (2-4)?

# In[19]:


#create a new dataframe of non null values in the phases column

cct_phases = cct_copy[(cct_copy['Phases'].notnull()) & (cct_copy['Phases'] != 'Not Applicable')]


# In[20]:


cct_phases.head()


# In[21]:


cct_phases['Phases'].value_counts()


# In[22]:


cct_phases['Status'].value_counts()


# In[23]:


cct_phases = cct_phases[cct_phases['Status'] == 'Completed']


# In[24]:


cct_phases


# ### What is the time frame of this dataset?
# 

# In[33]:


cct_phases.info()


# In[34]:


cct_phases['Start Year'] = cct_phases['Start Date'].dt.year


# In[35]:


cct_phases['Completion Year'] = cct_phases['Completion Date'].dt.year


# In[41]:


result = cct_phases[['Start Year', 'Completion Year']].agg(['min', 'max'])
result


# The clinical trials in this dataset started in 2013 and some ended in 2021. The dataframe needs to be analyzed further because it does not only contain covid related clinical trials since covid pandemic occured in 2019. 

# In[51]:


pd.set_option('display.max_rows', None)
cct_phases['Conditions'].value_counts()

##filter dataframe to include terms such as COVID-19, COVID, Corona Virus Infection, Coronavirus, COVID19,COVID 19, Covid19


# ### What phase completed the most covid clinical trials? 

# In[28]:


s = cct_phases['Phases'].value_counts()


# In[29]:


cct_phases_df = s.to_frame()


# In[30]:


cct_phases_df.reset_index().rename(columns={'index': 'Phases', 'Phases':'Number of Trials'})


# In[31]:


base_color = sb.color_palette()[0]

sb.countplot(data=cct_phases, x='Phases', color=base_color)
plt.xticks(rotation=70)
plt.title('The number of covid-related clinical trials in each of the phases(2-4)');


# The highest number of clinical trials of COVID completed were in Phase 2, where the participants receiving the intervention are compared to those receiving a placebo to determine the drug's effectiveness. 
# 
# #### What types of interventions were tested in clinical trials that completed Phase 2? 

# In[32]:


cct_phases['Study Type'].value_counts()


# All COVID related clinical trials marked completed are interventional study type. An interventional study type in clinical trial is when participants are randomly assigned to receive an intervention or treatment in order to evaluate its effects. The aim of this study type is to determine the efficacy and safety of the intervention being studied. 
# 
# Interventional studies are also known as randomized controlled trials (RCTs), the gold standard in clinical research. In an RCT, particpants are randomly assigned to either the treatment group or the control group. The treatment group receives the drug being tested and the control group received the placebo or standard treatment. Then the results of the two groups are compared to determine the effectiveness of the intervention. Interventional studies help to make sure that the new treatments are safe and effective before they are released to the public. 

# ### What are the top five sponsors?

# In[42]:


cct_phases['Sponsor/Collaborators'].value_counts()

