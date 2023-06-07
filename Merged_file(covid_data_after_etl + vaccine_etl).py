#!/usr/bin/env python
# coding: utf-8

# <font size=4><font color=red>Importing files

# In[1]:


import pandas as pd
covid_data = pd.read_csv('covid_data_after_etl.csv')
covid_data


# In[2]:


vaccine_data = pd.read_csv('vaccine_etl.csv')
vaccine_data


# <font size=4><font color=red>Left Join</font> of covid_data_after_etl + vaccine_etl

# In[3]:


# Perform the left join
merged_df = pd.merge(covid_data, vaccine_data, left_on='Date', right_on='vaccine_date', how='left')

# Display the merged data frame
merged_df


# <font size=3>Checking Info of Merged_df

# In[4]:


merged_df.info()


# <font size=4><font color=red>Droping duplicate column or unwanted column

# In[5]:


merged_df.drop(['State', 'Sessions'], inplace=True, axis=1)


# Verifying columns are dropped or not

# In[6]:


merged_df.info()


# Successfully droped columns 'State' and 'Sessions'

# <font size=4><font color=red>Storing</font> df_cleaned = merged_df

# In[7]:


df_cleaned = merged_df


# <font size=4><font color=red>Plot Line Chart between 'Total_Doses_Administered' Vs 'Confirmed' Vs 'Total_Individuals_Vaccinated' with respect to 'Date'
# <font size=3><font color=black><br>1. Group the DataFrame by 'Date' and calculated the sum for each column
# <br>2. Set the figure size
# <br>3. Plot each column as an individual line graph with different colors
# <br>4. Set the title and labels
# <br>5. Rotate the x-axis labels for better visibility
# <br>6. Display the legend
# <br>7. Display the chart

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Group the DataFrame by 'Date' and calculate the sum for each column
sum_df = df_cleaned.groupby('Date')[['Total_Doses_Administered', 'Confirmed', 'Total_Individuals_Vaccinated']].sum()

# Set the figure size
plt.figure(figsize=(9, 4))

# Plot each column as an individual line graph with different colors
ax = sum_df['Total_Doses_Administered'].plot(color='red', label='Total Doses Administered')
sum_df['Confirmed'].plot(color='green', label='Confirmed Cases')
sum_df['Total_Individuals_Vaccinated'].plot(color='blue', label='Total Individuals Vaccinated')

# Set the title and labels
plt.title('Total Doses Administered Vs Confirmed Cases Vs Total Individuals Vaccinated')
plt.xlabel('Date')
plt.ylabel('Value')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Display the legend
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()


# <font size=4><font color=green>Observation</font>
# <font size=3><font color=black><br>1. Official announcement covid_19 in India is 30 Jan 2020.
# <br>2. After one 11 months 15 days on 16 Jan 2021 India announces that they have made their covid vaccine. Till then confirmed covid cases had reached to 10.51 millions.
# <br>3. After that Indian Government where able to control covid cases in India

# <font size=4><font color=red>Plot LIne Chart between 'Cured' Vs 'Active_cases' Vs 'Deaths' with respect to 'Date'
# <font size=3><font color=black><br>1. Group the DataFrame by 'Date' and calculated the sum for each column
# <br>2. Set the figure size
# <br>3. Plot each column as an individual line graph with different colors
# <br>4. Set the title and labels
# <br>5. Rotate the x-axis labels for better visibility
# <br>6. Display the legend
# <br>7. Display the chart

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

# Group the DataFrame by 'Date' and calculate the sum for each column
sum_df = df_cleaned.groupby('Date')[['Cured', 'Active_cases', 'Deaths']].sum()

# Set the figure size
plt.figure(figsize=(9, 4))

# Plot each column as an individual line graph with different colors
ax = sum_df['Cured'].plot(color='red', label='Cured Cases')
sum_df['Active_cases'].plot(color='green', label='Active Cases')
sum_df['Deaths'].plot(color='blue', label='Deaths cases')

# Set the title and labels
plt.title('Cured Vs Active_cases Vs Deaths')
plt.xlabel('Date')
plt.ylabel('Value')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Display the legend
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()


# We can see that after invention of Covid Vaccines People started curing rapidly

# <font size=4><font color=red>Plot Area Chart between '_Covaxin_(Doses_Administered)' Vs 'CoviShield_(Doses_Administered)' Vs 'Sputnik_V_(Doses_Administered)' with respect to 'Date'
# <font size=3><font color=black><br>1. Group the DataFrame by 'Date' and calculated the sum for each column
# <br>2. Set the figure size
# <br>3. Plot each column as an individual line graph with different colors
# <br>4. Set the title and labels
# <br>5. Rotate the x-axis labels for better visibility
# <br>6. Adjusting y-axis limit
# <br>7. Display the legend
# <br>8. Display the chart

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt

# Group the DataFrame by 'Date' and calculate the sum for each column
sum_df = df_cleaned.groupby('Date')[['_Covaxin_(Doses_Administered)', 'CoviShield_(Doses_Administered)', 'Sputnik_V_(Doses_Administered)']].sum()

# Set the figure size
plt.figure(figsize=(9, 4))

# Plot each column as an individual area chart with different colors and custom labels
ax = sum_df['_Covaxin_(Doses_Administered)'].plot.area(color='red', label='Covaxin')
sum_df['CoviShield_(Doses_Administered)'].plot.area(color='green', label='CoviShield')
sum_df['Sputnik_V_(Doses_Administered)'].plot.area(color='blue', label='Sputnik V')

# Set the title and labels
plt.title('Covaxin vs CoviShield vs Sputnik V')
plt.xlabel('Date')
plt.ylabel('Value')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Adjust the y-axis limits
plt.ylim(0, sum_df.max().max() * 1.1)  # Increase the factor as needed

# Display the legend
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()


# <font size=4><font color=red>Plot Pie Chart between '_Covaxin_(Doses_Administered)' Vs 'CoviShield_(Doses_Administered)' Vs 'Sputnik_V_(Doses_Administered)'
# <font size=3><font color=black><br>1. Calculated the grand sum for each column
# <br>2. Set the figure size
# <br>3. Create a pie chart with the grand sum data
# <br>4. Set the title
# <br>5. Display the chart

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Calculate the grand sum of the columns
grand_sum = df_cleaned[['_Covaxin_(Doses_Administered)', 'CoviShield_(Doses_Administered)', 'Sputnik_V_(Doses_Administered)']].sum()

# Set the figure size
plt.figure(figsize=(4, 4))

# Create a pie chart with the grand sum data
plt.pie(grand_sum, labels=grand_sum.index, autopct='%1.5f%%', colors=['red', 'green', 'blue'])

# Set the title
plt.title('Vaccine Distribution')

# Display the chart
plt.tight_layout()
plt.show()


# <font size=4><font color=green>Observation</font>
# <font size=3><font color=black><br>1. Number one covid vaccine used by India is CoviShield with 88.5722% followed by Covaxin 11.386%.
# <br>2. Sputnik V is list adopted covid vaccine in India which was used only 0.04136%.

# <font size=5><font color=red>Saving/Downloading </font>df_cleaned as csv file as merged_cleaned file

# In[12]:


merged_cleaned = df_cleaned


# In[13]:


merged_cleaned.to_csv('merged_cleaned.csv',index=False)


# merged_cleaned.csv file saved successfully

# In[ ]:




