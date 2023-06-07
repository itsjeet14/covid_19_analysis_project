#!/usr/bin/env python
# coding: utf-8

# <font color=red><font size=5>Import pandas and covid_19_india.csv data.

# In[1]:


import pandas as pd
covid_data = pd.read_csv("covid_19_india.csv")
covid_data


# <font color=red><font size=5>Since the index is starting from 0, so i want to make it to start from 1 and so on.......

# In[2]:


covid_data.index = covid_data.index + 1
covid_data


# <font color=red><font size=5>Find top 10 rows.

# In[3]:


covid_data.head(10)


# <font color=red><font size=5>Find column details/properties.

# In[4]:


covid_data.info()


# <font color=red><font size=5>Find description of table.

# In[5]:


covid_data.describe()


# <font color=red><font size=5>Drop columns which are not required from covid_19_india.csv
# <br><font color=black><font size=3>As from the observation we have seen that there is no any use of Sno, Time, ConfirmedIndianNational and ConfirmedForeignNational in the further analysis thats why we are droping these columns.
# <br><font color=green><font size=3>Enter one time shift+Enter and the function will be performed because after that it will show required columns are not found.

# In[6]:


covid_data.drop(['Sno','Time','ConfirmedIndianNational','ConfirmedForeignNational'],inplace=True,axis=1)


# <font size=3>Check the column Info to know desired columns are removed or not?

# In[7]:


covid_data.info()


# <font color=red><font size=5>Change data type of Date column
# <br><font color=green><font size=3>Enter one time shift+Enter and the function will be performed because after that it will show required columns are not found.

# In[8]:


covid_data['Date'] = pd.to_datetime(covid_data['Date'],format = '%Y-%m-%d')


# <font size=3>Check the column Info to know Date column format changed or not?

# In[9]:


covid_data.info()


# <font size=3>Show the covid_data

# In[10]:


covid_data


# <font color=red><font size=5>Add a new column as Active_cases 

# In[11]:


covid_data['Active_cases'] = covid_data['Confirmed'] - (covid_data['Cured']+covid_data['Deaths'])
covid_data


# <font size=3>Checking the negative values Cured, Deaths, Confirmed, and Active_cases

# In[12]:


negative_Cured_count = sum(covid_data['Cured'] < 0)
negative_Deaths_count = sum(covid_data['Deaths'] < 0)
negative_Confirmed_count = sum(covid_data['Confirmed'] < 0)
negative_Active_cases_count = sum(covid_data['Active_cases'] < 0)
print("Negative Cured Count: ",negative_Cured_count)
print("Negative Deaths Count: ",negative_Deaths_count)
print("Negative Confirmed Count: ",negative_Confirmed_count)
print("Negative Active Cases: ",negative_Active_cases_count)


# <font size=3>Checking which State/UnionTerritory has negative Active_cases

# In[13]:


negative_rows = covid_data[covid_data['Active_cases'] < 0]
negative_rows


# <font size=3>Removing these rows having negative Active_cases because it can't be negative
# <br><font color=blue> Press only one time Shift+Enter

# In[14]:


covid_data.drop(negative_rows.index, inplace=True)


# <font size=3>Checking negative Active_cases rows are deleted or not

# In[15]:


negative_Cured_count = sum(covid_data['Cured'] < 0)
negative_Deaths_count = sum(covid_data['Deaths'] < 0)
negative_Confirmed_count = sum(covid_data['Confirmed'] < 0)
negative_Active_cases_count = sum(covid_data['Active_cases'] < 0)
print("Negative Cured Count: ",negative_Cured_count)
print("Negative Deaths Count: ",negative_Deaths_count)
print("Negative Confirmed Count: ",negative_Confirmed_count)
print("Negative Active Cases: ",negative_Active_cases_count)


# <font color=red><font size=5>Check the duplicate from the column 'State/UnionTerritory'

# In[16]:


covid_data_State_UnionTerritory_element = covid_data['State/UnionTerritory'].unique()
covid_data_State_UnionTerritory_element


# <font size=3>Arrange the above result in assending order

# In[17]:


covid_data_State_UnionTerritory_element.sort()
covid_data_State_UnionTerritory_element


# <font color=red><font size=5>Removing Duplicate rows from covid_data from column '	State/UnionTerritory'
# <br><font color=black><font size=3> The duplicates we have
# <br>1. Maharashtra*** & Maharashtra
# <br>2. Bihar**** & Bihar
# <br>3. Madhya Pradesh*** & Madhya Pradesh
# <br>4. Karanataka & Karnataka
# <br>5. Telengana & Telangana
# <br>6. Himanchal Pradesh & Himachal Pradesh

# In[18]:


covid_data['State/UnionTerritory']=covid_data['State/UnionTerritory'].replace('Maharashtra***','Maharashtra')
covid_data['State/UnionTerritory']=covid_data['State/UnionTerritory'].replace('Bihar****','Bihar')
covid_data['State/UnionTerritory']=covid_data['State/UnionTerritory'].replace('Madhya Pradesh***','Madhya Pradesh')
covid_data['State/UnionTerritory']=covid_data['State/UnionTerritory'].replace('Karanataka','Karnataka')
covid_data['State/UnionTerritory']=covid_data['State/UnionTerritory'].replace('Telengana','Telangana')
covid_data['State/UnionTerritory']=covid_data['State/UnionTerritory'].replace('Himanchal Pradesh','Himachal Pradesh')


# <font size=3>Verifying the duplicate elements in 'State/UnionTerritory' column

# In[19]:


covid_data_without_duplicates = covid_data['State/UnionTerritory'].unique()
covid_data_without_duplicates.sort()
covid_data_without_duplicates


# <font size=3>Count the unique element in 'State/UnionTerritory' column

# In[20]:


print("Number of unique elements in 'State/UnionTerritory' column after removing of duplicates:",len(covid_data_without_duplicates))


# <font size=4><font color=green>Removing rows 'Cases being reassigned to states' and 'Unassigned'

# <font size=3>Here, we can observe that 'Cases being reassigned to states' and 'Unassigned' rows do not belong to any  'State/UnionTerritory'. So I am removing these rows.

# <font size=3>Storing covid_data as covid_data_column_etl

# In[21]:


covid_data_after_column_etl = covid_data
covid_data_after_column_etl


# <font color=blue>Counting number of 'Unassigned' rows

# In[22]:


count_unassigned = len(covid_data_after_column_etl[covid_data_after_column_etl['State/UnionTerritory']=='Unassigned'])
count_unassigned


# <font size=3>Checking percentage of 'Unassigned' row data

# In[23]:


3/18110 *100


# <font size=3>Since out of 18110 rows its just 3 rows and 0.0165% of overall data so i am droping this rows.

# In[24]:


covid_data_after_column_etl = covid_data_after_column_etl.drop(covid_data_after_column_etl[covid_data_after_column_etl['State/UnionTerritory'] == 'Unassigned'].index)


# <font color=blue>Counting number of 'Cases being reassigned to states' rows

# In[25]:


count_Cases_being_reassigned_to_states = len(covid_data_after_column_etl[covid_data_after_column_etl['State/UnionTerritory']=='Cases being reassigned to states'])
count_Cases_being_reassigned_to_states


# <font size=3>Checking percentage of 'Cases being reassigned to states' row data

# In[26]:


60/18110 *100


# <font size=3>Since out of 18110 rows its just 60 rows and 0.33% of overall data so i am droping this rows.

# In[27]:


covid_data_after_column_etl = covid_data_after_column_etl.drop(covid_data_after_column_etl[covid_data_after_column_etl['State/UnionTerritory']=='Cases being reassigned to states'].index)


# <font size=3> Verifying element are removed or not from the State/UnionTerritory Column

# In[28]:


a = covid_data_after_column_etl['State/UnionTerritory'].unique()
a.sort()
a


# <font color=Blue><font size=3>Storing covid_data_column_etl = covid_data_after_etl

# In[29]:


covid_data_after_etl = covid_data_after_column_etl


# <font color=red><font size=5>Create a Pivot Table to Analyze COVID-19 Statistics by State/UnionTerritory

# <font size=3>COVID-19 Statewise Data Summary: Maximum 'Confirmed' Cases, 'Deaths', and 'Cured'

# In[30]:


statewise_data = pd.pivot_table(covid_data_after_etl,values=['Confirmed','Deaths',"Cured"],index='State/UnionTerritory', aggfunc=max)


# <font size=3>Calculating <font color=red>"Recovery Rate"<font color=black> for COVID-19 Statewise Data

# In[31]:


statewise_data['Recovery Rate'] = (statewise_data['Cured']/statewise_data['Confirmed'])*100


# <font size=3>Calculating <font color=red>"Mortality Rate"<font color=black> for COVID-19 Statewise Data

# In[32]:


statewise_data['Mortality Rate'] = (statewise_data['Deaths']/statewise_data['Confirmed'])*100


# <font size=3>Sorting COVID-19 Statewise Data by Confirmed Cases in Descending Order

# In[33]:


statewise_data = statewise_data.sort_values(by='Confirmed', ascending=False)


# <font size=3>Applying Color Gradient to COVID-19 Statewise Data Based on Confirmed Cases

# In[34]:


statewise_data.style.background_gradient(cmap='CMRmap')


# <font size=3><font color=green>Following observations, we can get from here👆👆 and all these observations are for a single day
# <br><font color=black>1. Maximum Confirmed covid cases --> Maharashtra & Lowest Confirmed covid cases --> Daman & Diu
# <br>2. Maximum Deaths --> Maharastra & Minimum Deaths --> Daman & Diu
# <br>3. Highest Recovery Rate --> Dadra and Nagar Haveli and Daman and Diu & Lowest Recovery Rate --> Daman & Diu
# <br>4. Maximum Mortality Rate --> Punjab & Minimum Mortality Rate --> Daman & Diu

# <Font size=4><font color=Blue>Top 10 active_cases States

# <font size=3>Top 10 active_cases States/UnionTerritory sorted by Active_cases in descending order

# In[35]:


top10ActiveCases=covid_data_after_etl.groupby(by='State/UnionTerritory').max()[['Active_cases','Date']].sort_values(by=['Active_cases'],ascending=False).reset_index()
top10ActiveCases.head(10)


# <font size=3>Plot top 10 active_cases states data in column chart<font color=green>
# <br>1. Size of graph --> 16*10
# <br>2. Title of chart --> 'Top 10 states with most Active Cases in India' & Text size --> 22
# <br>3. y-axis -->'Active_cases' & x-axis-->'State/UnionTerritory' & linewidth=2 & edgecolor='black'
# <br>4. x-label --> "States"
# <br>5. y-label --> "Total Active Cases"

# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
fig=plt.figure(figsize=(16,10))
plt.title("Top 10 states with most Active Cases  in India",size=22)
ax=sns.barplot(data=top10ActiveCases.iloc[:10],y='Active_cases',x='State/UnionTerritory',linewidth=2,edgecolor='black')
plt.xlabel("States")
plt.ylabel("Total Active Cases")
plt.show()


# <Font size=4><font color=Blue>Top 10 States with Highest Number of Deaths

# <font size=3>Top 10 States/UnionTerritory with Highest Number of Deaths sorted by Deaths in descending order

# In[37]:


top10Deaths = covid_data_after_etl.groupby(by='State/UnionTerritory').max()[['Deaths','Date']].sort_values(by=['Deaths'],ascending=False).reset_index()
top10Deaths.head(10)


# <font color=black>Plot top 10 states with Highest Number of Deaths in column chart<font color=green>
# <br>1. Size of graph --> 16*10
# <br>2. Title of chart --> 'Top 10 states with most Deaths in India' & Text size --> 22
# <br>3. y-axis -->'Deaths' & x-axis-->'State/UnionTerritory' & linewidth=2 & edgecolor='black'
# <br>4. x-label --> "States"
# <br>5. y-label --> "Total Deaths"

# In[38]:


fig = plt.figure(figsize=(16,10))
plt.title("Top 10 states with most Deaths in India",size=22)
ax = sns.barplot(data=top10Deaths.iloc[:10],y='Deaths',x='State/UnionTerritory',linewidth=2, edgecolor='black')
plt.ylabel("Total Deaths")
plt.xlabel('States')
plt.show()


# <font size=5><font color=black>Plot Growth Trend in lineplot for Top 5 affected States in India<font size=3><font color=green>
# <br>1. Size of graph --> 12,6
# <br>2. Title of chart --> 'Top 5 Affected States in India' & Text size --> 16
# <br>3. y-axis -->'Active_cases' & x-axis-->'Date' & hue --> 'State/UnionTerritory'

# In[39]:


fig = plt.figure(figsize=(12,6))
plt.title('Top 5 Affected States in India',size=16)
ax = sns.lineplot(data=covid_data_after_etl[covid_data_after_etl['State/UnionTerritory'].isin(['Maharashtra','Karnataka','Kerala','Tamil Nadu','Uttar Pradesh'])],x='Date',y='Active_cases',hue='State/UnionTerritory')


# <font size=5><font color=green>Conclusion<font color=black><font size=3>
# <br>1. Covid wave was cyclic of 4 months (1st four months it increased and for next 4 months it decreased)
# <br>2. Most affected month of 2020 was 2020-07 to 2020-11
# <br>3. Most affected month of 2021 was 2021-03 to 2021-07
# <br>4. Most affected state was Maharashtra then Karnataka, Tamil Nadu, Delhi, Utter Pradesh
# <br>5. Least affected state was Daman and Diu

# <font size=5><font color=red>Saving/Downloading </font>covid_data_after_etl as csv file

# In[40]:


# Save it to a CSV file
covid_data_after_etl.to_csv('covid_data_after_etl.csv', index=False)


# <font size=3>Check the current working directory

# In[41]:


import os
print(os.getcwd())


# In[ ]:





# In[ ]:




