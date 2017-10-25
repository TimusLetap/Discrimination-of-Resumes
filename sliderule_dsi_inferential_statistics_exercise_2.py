
# coding: utf-8

# # Examining Racial Discrimination in the US Job Market
# 
# ### Background
# Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.
# 
# ### Data
# In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.
# 
# Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

# <div class="span5 alert alert-info">
# ### Exercises
# You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.
# 
# Answer the following questions **in this notebook below and submit to your Github account**. 
# 
#    1. What test is appropriate for this problem? Does CLT apply?
#    2. What are the null and alternate hypotheses?
#    3. Compute margin of error, confidence interval, and p-value.
#    4. Write a story describing the statistical significance in the context or the original problem.
#    5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?
# 
# You can include written notes in notebook cells using Markdown: 
#    - In the control panel at the top, choose Cell > Cell Type > Markdown
#    - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# 
# #### Resources
# + Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
# + Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# </div>
# ****

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
import seaborn as sns
import math
sns.set()


# In[2]:

data = pd.io.stata.read_stata('us_job_market_discrimination.dta')


# In[3]:

# number of callbacks for black-sounding names
sum(data[data.race=='b'].call)


# In[4]:

# number of callbacks for white-sounding names
sum(data[data.race=='w'].call)


# In[5]:

data.head()


# In[6]:

data['id'].describe()


# In[7]:

list(data)


# In[8]:

df= data[['race', 'call']]
df.head()


# In[9]:

freq= df.pivot(columns= 'race', values= 'call')
freq.head()


# In[10]:

freq.describe()


# In[11]:

df.groupby(["race", "call"]).size()


# In[12]:

pd.crosstab(df.race, df.call)


# ## Generate initial visualizations for preliminary observations

# In[70]:

_= df.call.value_counts('w').plot(kind='bar')
plt.show()


# In[130]:

_= df.call.value_counts('b').plot(kind='bar')
plt.show()


# * initial observation with histograms shows very little to no diffrence between callback by value counts of b and w

# In[55]:

_= df.groupby([df['race'], df['call']]).size().plot(kind='bar')
_= plt.xlabel('Race')
_= plt.ylabel('Callback')
_= plt.title('Callback by Race')
plt.show()


# * plotting a histogram with groupby provides a completely different observation 
# * we can see from this that there is a significant observable difference between callback values of b and w

# In[18]:

def ecdf(df):
    
    # Number of data points: n
    n = len(df)

    # x-data for the ECDF: x
    x = np.sort(df)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# ## Identifying the problem 
# * Does the proportion of callbacks differ between black and white resumes?
# * CLT applies even though the there is a binomial distribution of population the sampling proportions will be normal

# In[101]:

# Assign variable for each race from dataset 
all_b = df[df['race'] == 'b']
all_w = df[df['race'] == 'w']

# Sample size by race
l_b= len(all_b)
l_w= len(all_w)
print(l_b)
print(l_w)

# Samples that received callback
cb_b= all_b.call.sum()
cb_w= all_w.call.sum()

# Ratio of call back 
P_b= cb_b/l_b
P_w= cb_w/l_w


[P_b, P_w]


# * since both sample sizes are the same we will only need to use one for further calculations
# * from this we can see that approximately 9.7% of white sounding names received callbacks whereas approximately 6.4% of black sounding names received callbacks
# * we need to use the 2-sample t-test to determine if the two samples (b and w) are different using the standard 5% threshhold

# In[128]:

# Variance
var_b= np.var(all_b)
var_w= np.var(all_w)

# Difference of means

mu_diff= (P_w - P_b)


#Standard deviation
std_b= np.std(all_b)
std_w= np.std(all_w)

# Difference of Standard Deviations of sampling distributions
std_diff= math.sqrt((var_w/l_w)+(var_b/l_b))

# Critical t_value for 95%

t_crit= stats.t.ppf(0.975, l_b-1)
print(t_crit)

# Margin of Error
d= t_crit*std_diff

# 95% confidence interval
CI= [mu_diff-d, mu_diff+d]
CI



# In[124]:

# T-Score
t_score= mu_diff/(std_diff)
t_score


# In[125]:

# P-Value

p= stats.t.sf(t_score, l_b-1)*2
p


# In[127]:

stats.ttest_ind(all_w.call,all_b.call,equal_var=False)


# # Conclusion
# 
# * observed p-value is very small so null hypothesis can be rejected and that there is a significant diffrence in callbacks between samples (b and w)
# * the 95% confidence interval does not contain a 0 which allows us to conclude that there is a significant diffrence in the means
