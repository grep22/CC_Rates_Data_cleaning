#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:52:03 2019

@author: GuiReple
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_OG = pd.read_csv('loan_interest_rates.csv', header=0)

###############################################################################
#############################CLEANING DATA#####################################
###############################################################################


#############Renaming Columns according to Metadata############################

df = df_OG.rename(columns={'X1': 'int_rate_loan',
                        'X2': 'loan_id',
                        'X3': 'borrower_id',
                        'X4': 'loan_amount_request',
                        'X5': 'loan_funded',
                        'X6': 'inv_portion_fund',
                        'X7': 'num_payments',
                        'X8': 'loan_grade',
                        'X9': 'loan_subgrade',
                        'X10': 'employer_job_title',
                        'X11': 'years_employed',
                        'X12': 'home_owner_status',
                        'X13': 'year_income',
                        'X14': 'verified_income',
                        'X15': 'date_loan_issued',
                        'X16': 'reason_from_borrower',
                        'X17': 'loan_category',
                        'X18': 'loan_title',
                        'X19': 'state_borrower',
                        'X20': 'debt_to_income',
                        'X21': 'num_30days_past_due',
                        'X22': 'date_report_credit_open',
                        'X23': 'num_month_last_delinquency',
                        'X24': 'num_months_last_pub_record',
                        'X25': 'num_derogatory_pub_record',
                        'X26': 'num_credit_lines_file',
                        'X27': 'initial_stats_loan'})

###############################################################################
#############################Count Missing Values##############################

df.info()
df_miss_list = df.isnull().sum()

"""
MISSING VALUES SUMMARY

int_rate_loan	61010
loan_id	1
borrower_id	1
loan_amount_request	1
loan_funded	1
inv_portion_fund	1
num_payments	1
loan_grade	61270
job_subgrade	61270
employer_job_title	23986
years_employed	17538
Home_owner_status	61361
year_income	61028
verified_income	1
date_loan_issued	1
reason_from_borrower	276440
loan_category	1
loan_title	19
state_borrower	1
ratio_borrowed_vs_obligation	1
num_30days_past_due	1
date_report_credit_open	1
num_month_last_delinquency	218802
num_months_last_pub_record	348845
num_derogatory_pub_record	1
num_credit_lines_file	1
initial_stats_loan	1

"""
"""Main issues here is that int_rate_loan has ~15% of missing value. 
Other issues:
    -employer_job_title is user fill-in so there is no structure and will be dificult
    to analyze. If had more time would try and run an analysis with text analytics.
    -home_owner_status: many missing values. Will substitute them for -1 or none.
    -year_income: many missing values. Also many extreme values ranging from 500k-7.4million.
                Its difficult to believe that people who make over 7.4 million a year need a 10k
                to 35k loan - unless they are trying to deduct taxes. 
    -reason_from_borrower: pure user-input text. Would require text analytics with more time
                to go through analysis. Maybe understand user sentiment when explaining reason
                for taking out loan. 
    -loan_title: user-input, some very random names others descriptive.
    -num_month_last_delinquency/num_months_last_pub_record: 70-85% missing data. This will not 
                be used in this analysis. Too many missing values. Would be amazing to have this
                in order to run a predicting model on interest rates. 
    -NOTE: it seems that one entry doesn't have any value. """


###############################################################################
#################Converting strings to floats or int###########################
###############################################################################

#####int_rate_loan = cleaning column to become float for analysis
df['int_rate_loan'] = df['int_rate_loan'].str.rstrip('%').astype('float') / 100.0

##plt.hist(df['int_rate_loan'], bins=20)
"""int_rate_loan seems to be almost normally distributed with a slight right skew."""


#####loan_amount_request = cleaning column to become float for analysis
df['loan_amount_request'] = df.loan_amount_request.str.replace(",","")
df['loan_amount_request'] = df.loan_amount_request.str.replace("$","")
df['loan_amount_request'] = df['loan_amount_request'].astype(float)

##plt.hist(df['loan_amount_request'],bins = 10)
"""loan_amount_request seems to be normally distributed with a slight right skew."""


#####loan_funded = cleaning column to become float for analysis
df['loan_funded'] = df.loan_funded.str.replace(",","")
df['loan_funded'] = df.loan_funded.str.replace("$","")
df['loan_funded'] = df['loan_funded'].astype(float)


#####inv_portion_fund = cleaning column to become float for analysis
df['inv_portion_fund'] = df.inv_portion_fund.str.replace(",","")
df['inv_portion_fund'] = df.inv_portion_fund.str.replace("$","")
df['inv_portion_fund'] = df['inv_portion_fund'].astype(float)


#####num_payments = creating new column with only int for analysis
df['num_payments_int'] = df['num_payments'].str.rstrip('months')
df['num_payments_int'] = df['num_payments_int'].astype(float)


#####loan_subgrade = eliminate Letter for analysis
df['loan_subgrade_int'] = 0
df['loan_subgrade_int']= df['loan_subgrade'].str[1]
#df['loan_subgrade_int'] = df['loan_subgrade_int'].astype(int)


##### years_employed converted to int for easier call on code. 
df['years_employed'].value_counts()
"""
10+ years    128060
2 years       35427
3 years       31428
< 1 year      30607
5 years       27277
1 year        25321
4 years       24035
6 years       23062
7 years       22837
8 years       19139
9 years       15269

***Most people claim to have more than 10 years of work experience. 
"""

years_employed_dictionary = {'< 1 year':0,
                             '1 year':1,
                             '2 years':2,
                             '3 years':3,
                             '4 years':4,
                             '5 years':5,
                             '6 years':6,
                             '7 years':7,
                             '8 years':8,
                             '9 years':9,
                             '10+ years':10}

df['years_employed_cat'] = df['years_employed']
df['years_employed_cat'].replace(years_employed_dictionary, inplace = True)
##df['years_employed_cat'].astype(int)
df['years_employed_cat'].value_counts()
df['years_employed_cat'].isnull().sum()


################################ home_owner_status
df['home_owner_status'].value_counts()

###########Creating columns for dummy variables
df['hos_mortgage']=0
df['hos_rent']=0
df['hos_own']=0
df['hos_other']=0
df['hos_none'] = 0
df['hos_any']=0

##################Setting where conditions for Dummy Variables
df['hos_mortgage'] = np.where((df.home_owner_status == "MORTGAGE"), "1","0")
df['hos_mortgage'] = df['hos_mortgage'].astype(int)

df['hos_rent'] = np.where((df.home_owner_status == "RENT"), "1","0")
df['hos_rent'] = df['hos_rent'].astype(int)

df['hos_own'] = np.where((df.home_owner_status == "OWN"), "1","0")
df['hos_own'] = df['hos_own'].astype(int)

df['hos_other'] = np.where((df.home_owner_status == "OTHER"), "1","0")
df['hos_other'] = df['hos_other'].astype(int)

df['hos_none'] = np.where((df.home_owner_status == "NONE"), "1","0")
df['hos_none'] = df['hos_none'].astype(int)

df['hos_any'] = np.where((df.home_owner_status == "ANY"), "1","0")
df['hos_any'] = df['hos_any'].astype(int)


##### loan_category - 
df['loan_category'].value_counts()

df['loantype_debt_consolidation'] = 0 
df['loantype_debt_consolidation'] = np.where((df.loan_category == "debt_consolidation"), "1","0")
df['loantype_debt_consolidation'] = df['loantype_debt_consolidation'].astype(int)
   
df['loantype_credit_card'] = 0 
df['loantype_credit_card'] = np.where((df.loan_category == "credit_card"), "1","0")
df['loantype_credit_card'] = df['loantype_credit_card'].astype(int)
          
df['loantype_home_improvement'] = 0 
df['loantype_home_improvement'] = np.where((df.loan_category == "home_improvement"), "1","0")
df['loantype_home_improvement'] = df['loantype_home_improvement'].astype(int)
      
df['loantype_other'] = 0  
df['loantype_other'] = np.where((df.loan_category == "other"), "1","0")
df['loantype_other'] = df['loantype_other'].astype(int)
                
df['loantype_major_purchase'] = 0
df['loantype_major_purchase'] = np.where((df.loan_category == "major_purchase"), "1","0")
df['loantype_major_purchase'] = df['loantype_major_purchase'].astype(int)
          
df['loantype_small_business'] = 0          
df['loantype_small_business'] = np.where((df.loan_category == "small_business"), "1","0")
df['loantype_small_business'] = df['loantype_small_business'].astype(int)

df['loantype_car'] = 0                     
df['loantype_car'] = np.where((df.loan_category == "car"), "1","0")
df['loantype_car'] = df['loantype_car'].astype(int)

df['loantype_medical'] = 0  
df['loantype_medical'] = np.where((df.loan_category == "medical"), "1","0")
df['loantype_medical'] = df['loantype_medical'].astype(int)
               
df['loantype_moving'] = 0 
df['loantype_moving'] = np.where((df.loan_category == "moving"), "1","0")
df['loantype_moving'] = df['loantype_moving'].astype(int)
                 
df['loantype_wedding'] = 0  
df['loantype_wedding'] = np.where((df.loan_category == "wedding"), "1","0")
df['loantype_wedding'] = df['loantype_wedding'].astype(int)
                
df['loantype_vacation'] = 0 
df['loantype_vacation'] = np.where((df.loan_category == "vacation"), "1","0")
df['loantype_vacation'] = df['loantype_vacation'].astype(int)
               
df['loantype_house'] = 0 
df['loantype_house'] = np.where((df.loan_category == "house"), "1","0")
df['loantype_house'] = df['loantype_house'].astype(int)
               
df['loantype_educational'] = 0  
df['loantype_educational'] = np.where((df.loan_category == "educational"), "1","0")
df['loantype_educational'] = df['loantype_educational'].astype(int)
            
df['loantype_renewable_energy'] = 0 
df['loantype_renewable_energy'] = np.where((df.loan_category == "renewable_energy"), "1","0")
df['loantype_renewable_energy'] = df['loantype_renewable_energy'].astype(int)         


##############################date_report_credit_open
"""date_report_credit_open has 2 types of data format MM-YY or YY-MM. Those years 
before the 90's are MM-YY, and those after 2001 are YY-MM. Had to work on this a little
differently to be able to compare recent opened accounts with old open accounts. Unfortunately
this doesn't tell us the age of the person who opened up credit, but it gives us an idea
of time with credit. Will do a comparisson between both in Tableau. """

df['date_report_credit_open'].value_counts()

####date 45-99
df['year_credit_opened'] = 0
df['year_credit_opened']= df['date_report_credit_open'].str[4:6]

####date 2001-2010
df['recent_open_credit']=0
df['recent_open_credit']= df['date_report_credit_open'].str[0:2]
df['recent_open_credit'].value_counts()

recent = {'Ja': 0,
       'Fe':0,
       'Ma':0,
       'Ap':0,
       'Ju':0,
       'Au':0,
       'Se':0,
       'Oc':0,
       'No':0,
       'De':0,
       '1-':2001,
       '2-':2002,
       '3-':2003,
       '4-':2004,
       '5-':2005,
       '6-':2006,
       '7-':2007,
       '8-':2008,
       '9-':2009,
       '10':2010,
       '11':2011
       }
df['recent_open_credit'].replace(recent,inplace=True)



df['year_credit_opened'].value_counts()


###repl represents people who have opened up their accounts from 2001-2010
repl = {'n': '01',
        'r': '01',
        'p': '01',
        'g': '01',
        't': '01',
        'l': '01',
        'v': '01',
        'y': '01',
        'c': '01',
        'b': '01',
        'an':'01',
        'ar': '01',
        'pr': '01',
        'eb': '01',
        'ep': '01',
        'ug': '01',
        'ay': '01',
        'un': '01',
        'ul': '01',
        'ct': '01',
        'ov': '01',
        'ec': '01'       
        }

df['year_credit_opened'].replace(repl, inplace = True)

###### initial_stats_loan W=1 and F=0
df['initial_stats_loan_w']= df['initial_stats_loan']
def func(x):
    if x == 'w':
        return 1
    else:
        return 0 
df['initial_stats_loan_w'].value_counts()

df['initial_stats_loan_w'] = df['initial_stats_loan_w'].map(func)


df.info()

df_stats = df.describe(include = 'all')

###############################################################################
#######################FEATURE ENGINEERING#####################################
###############################################################################
"""Objective of this is to get more data points from yearly income and interest rates
and payback period to further analysis."""

#######debt_to_income adjust to percentage

#######Monthly Salary
df['monthly_income'] = df['year_income']/12

#######debt amount according to ratio in X20(debt_to_income)
df['monthly_debt'] = df['monthly_income'] / df['debt_to_income']


#######fully funded vs not fully funded loan
df['full_funded'] = 0

df['fully_funded'] = np.where((df.loan_amount_request == df.loan_funded), "1","0")
df['loantype_educational'] = df['loantype_educational'].astype(int)

df['fully_funded'].value_counts()


#####calculating total loan to pay + monthly payment + amount earned from interest
### A = P(1 + rt)

df['total_loan_topay'] = 0
df['total_loan_topay'] = (df['loan_funded']*(1+(df['int_rate_loan']*(df['num_payments_int']/12))))


df['rough_amount_monthly']=0
df['rough_amount_monthly'] = df['total_loan_topay']/df['num_payments_int']


df['total_interest'] = 0
df['total_interest'] = df['total_loan_topay'] - df['loan_funded']
df['total_interest'].sum()


################Seperate entries into 4 groups. In Stats Description Income is
########divided in 25%-45000.0/ 50%-63000.0 / 75%-88200.0. Will seperate ranges.
######## These ranges will be seperated into Low, Mid, High, Million 
"""When running the loop below, I encountered time constraint and had to shut it down.
I tried another quicker way to get the categorical data translated"""

df['low_income_group']=0
df['low_income_group'] = np.where((df['year_income'] <= 45000), 1,0)
df['mid_income_group']=0
df['mid_income_group'] = np.where(np.logical_and(
                df['year_income'] >= 45001, df['year_income'] <=63000),1,0)
df['high_income_group']=0
df['high_income_group'] =np.where(np.logical_and(
                df['year_income'] >= 63001, df['year_income'] <=88200),1,0)
df['mil_income_group']=0
df['mil_income_group']= np.where((
                df['year_income'] >= 88201),1,0)

df['income_group']=0

next(df.iterrows())

row = next(df.iterrows())


####This for loop will take a while to run
for i, row in df.iterrows():
    if (df.loc[i,'low_income_group'] ==1):
        df.loc[i,'income_group'] = 'low_income'
        
    elif (df.loc[i,'mid_income_group'] ==1):
        df.loc[i,'income_group'] = 'mid_income'
        
    elif (df.loc[i,'high_income_group'] ==1):
        df.loc[i,'income_group'] = 'high_income'
        
    elif (df.loc[i,'mil_income_group'] ==1):
        df.loc[i,'income_group'] = 'mil_income'
        
    else:
        df.loc[i,'income_group'] = 'none'
        
df.income_group.value_counts()

###############################################################################
###########################EXPLORATORY ANALYSIS################################
################PRE MISSING VALUE INPUT TO ANALYSE FREQUENCY###################
###############################################################################
"""For data analyst purpose I will be substituting missing values for -1 to flag them and
not have them included in my analysis. I want to work for this particular project with data
that we know being that out of 400k observations we still have a very robust sample size not
considering the values that are null. If this were a ML problem I would be imputing missing
values according to median in different strategy scenarios. """

df_miss = df.copy()

fill = -1

df_miss = df_miss.fillna(fill)
df_miss.isnull().sum()


##############FREQUENCY DISTRIBUTION WITH -1 IMPLEMENTED ######################
############## DATA EXPLORATION VERY BASIC BEFORE MOVING ######################
################## TO WORKING WITH THE DATA IN TABLEAU ########################
plt.hist(df_miss['int_rate_loan'], bins=25)

plt.hist(df_miss['loan_amount_request'], bins=15)

plt.hist(df_miss['year_income'],bins= 5)

plt.hist(df_miss['num_credit_lines_file'])

plt.hist(df_miss['debt_to_income'])

plt.hist(df_miss['num_payments_int'])

plt.hist(df_miss['years_employed_cat'])

plt.hist(df_miss['year_credit_opened'])

count_date_ = df_miss['date_report_credit_open'].value_counts()

###############################################################################
###########Saving to csv to explore with Tableau or Excel######################
###############################################################################

#df_miss.to_csv("data_explore_miss.csv") #Didn't use this one for data exploration.
df.to_csv("data_explore_added.csv")


df_stats = df.describe(include = 'all')



df['loan_category'].value_counts()

df['year_credit_opened'].value_counts()
df_miss['recent_open_credit'].value_counts()




