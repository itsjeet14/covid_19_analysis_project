#!/usr/bin/env python
# coding: utf-8

# <font color=red><font size=5>Import covid_vaccine_statewise.csv file.

# In[1]:


import pandas as pd


# In[2]:


covid_vaccine_data = pd.read_csv("covid_vaccine_statewise.csv")
covid_vaccine_data


# <font color=red><font size=5>Find top 10 rows.

# In[3]:


covid_vaccine_data.head(10)


# <font color=red><font size=5>Find column details/properties.

# In[4]:


covid_vaccine_data.info()


# <font size=3><font color=green> Changing data type of 'Updated On' column from 'object' to 'date'

# In[5]:


covid_vaccine_data['Updated On'] = pd.to_datetime(covid_vaccine_data['Updated On'], format='%d/%m/%Y')
covid_vaccine_data.info()


# <font size=3>Replace spaces with underscores in column names

# In[6]:


new_columns = [col.replace(' ', '_') for col in covid_vaccine_data.columns]
covid_vaccine_data.columns = new_columns


# <font color=red><font size=5>Find description of table.

# In[7]:


covid_vaccine_data.describe()


# <font size=5><font color=red>ETL (Data cleaning)

# <font size=4><font color=Green>Change column name 'Updated On' to 'vaccine_date' 

# In[8]:


covid_vaccine_data.rename(columns={'Updated_On':'vaccine_date'},inplace=True)
covid_vaccine_data


# <font size=3>Here we can observe that under 'State' column there is 'India' as state.

# <font size=3>Find count of 'India' element in 'State' column

# In[9]:


india_count = len(covid_vaccine_data[covid_vaccine_data['State'] == 'India'])
india_count


# <font size=3>Percentage of 'india_count' with respect of 'Total rows'

# In[10]:


print('india_count: ',format(212/7845*100,'0.2f'))


# <font size=3><size color=green>Since this is just 2.7% so i am droping this column because 'India' does not comes under state.

# In[11]:


covid_vaccine_data_without_india = covid_vaccine_data[covid_vaccine_data['State'] != 'India']
covid_vaccine_data_without_india


# <font size=3>Verifying that rows containing 'India' is removed or not.

# In[12]:


india_count0 = len(covid_vaccine_data_without_india[covid_vaccine_data_without_india['vaccine_date'] == 'India'])
india_count0


# <font size=4><font color=red>Droping rows after 9 August 2021

# <font size=3>Checking last 20 rows of covid_vaccine_data_without_india

# In[13]:


covid_vaccine_data_without_india.tail(20)


# <br><font size=3><font color=black>On analysing the data from Power BI and Excel, I found that after 9 August 2021 all columns are empty so i am deleting rows after 9 August 2021

# In[14]:


covid_vaccine_data_without_india = covid_vaccine_data_without_india[covid_vaccine_data_without_india['vaccine_date'] <= '2021-08-09']
covid_vaccine_data_without_india


# <font size=4><font color=green>Visualisation of null values<font color=black>
# <br>1. Load the dataset
# <br>2. Display the missing value matrix for all columns
# <br>3. Show the plot

# In[15]:


import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
train = covid_vaccine_data_without_india

# Display the missing value matrix for all columns
msno.matrix(train)

# Show the plot
plt.show()


# 1. The plot appears blank(white) wherever there are missing values. For instance, in Embarked column there are only two instances of missing data, hence the two white lines.
# 
# 2. The sparkline on the right gives an idea of the general shape of the completeness of the data and points out the row with the minimum nullities and the total number of columns in a given dataset, at the bottom.

# <font size=4><font color=Green>Find null values in columns

# In[16]:


covid_vaccine_data_without_india.isnull().sum()


# <font size=4><font color=Green>Arranging null values in ascending order

# In[17]:


null_counts = covid_vaccine_data_without_india.isnull().sum()
null_counts_sorted = null_counts.sort_values(ascending=True)
null_counts_sorted


# <font size=3>We can observe that 👆👆 appart from column 'vaccine_date' and 'State', every column has null values

# <font size=4><font color=red>Handling Missing Value

# <font size=5><font color=green>===============================================================

# <font size=3><font color=black>Finding null value % so that we can we can decide which column we can keep and which column we can drop

# In[18]:


print("1. Covaxin (Doses Administered)")
print("2. Transgender (Doses Administered)")
print("3. Female (Doses Administered)")
print("4. Male (Doses Administered)")
print("5. CoviShield (Doses Administered) ")
print('6. First Dose Administered ')
print('7. Sites')
print("8. Sessions")
print("9. Total Doses Administered ")
print("10. Second Dose Administered )")

print("Null value %: ",format(1/7416*100,'0.2f'))


# <font size=3>Applying <font color=red>Skewness Method</font> to decide, How to fill missing values
# <br>1. Mean-It is preferred if data is numeric and not skewed.
# <br>2. Median-It is preferred if data is numeric and skewed.
# <br>3. Mode-It is preferred if the data is a string(object) or numeric.

# <font size=3>Calculating Skewness ignoring missing value i.e. particular row has missing values but this algorithm will bypass this.

# In[19]:


import pandas as pd
from scipy.stats import skew

data = covid_vaccine_data_without_india['Total_Doses_Administered']

# Calculate skewness, ignoring missing values
skewness = skew(data, nan_policy='omit')

print("Skewness:", format(skewness,"0.2f"))


# <font size=4><font color=green>Ploting Skewness Histogram with Bell curve</font><font size=3>
# <br>1. Calculate mean and standard deviation
# <br>2. Generate data points for the bell curve or Normal Distribution i.e. x-axis and y-axis
# <br>3. Plotting the bell curve and histogram
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. Use Histogram details as bins --> auto,desity, alpha --> 0.7, rwidth --> 0.85
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. Plot graph whith linewidth = 2   
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c. Histogram grid details as grid along x-axis, alpha --> 0.7
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d. xlabel --> value, ylabel --> Density
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e. Title --> Skewness Histogram with Bell Curve with decimal round up to 2 digits
# <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;f. legend --> Bell curve, Histogram

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Assuming you have a dataframe called "vaccine" with a column named "covid_vaccine"
data = covid_vaccine_data_without_india['Total_Doses_Administered']

# Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data)

# Generate data points for the bell curve
x = np.linspace(mean - 3*std, mean + 3*std, 100)
y = norm.pdf(x, mean, std)

# Plotting the bell curve and histogram
plt.hist(data, bins='auto', density=True, alpha=0.7, rwidth=0.85)
plt.plot(x, y, 'r-', linewidth=2)
plt.grid(axis='y', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Bell Curve with Histogram')
plt.legend(['Bell Curve', 'Histogram'])
plt.show()


# <font size=4>👆👆 We can observe that skewness = 2.46. It represents that these value are Right Skewed Distribution.<font size=3>
# <br>So I should replace the missing value with meadian 
# <br>_Covaxin_(Doses_Administered), Transgender_(Doses_Administered), Female_(Doses_Administered), Male_(Doses_Administered), CoviShield_(Doses_Administered), First_Dose_Administered, _Sites_, Sessions, Total_Doses_Administered, Second_Dose_Administered

# <font size=4>Stored<font color=green> covid_vaccine_data_without_india </font>as <font color=green>vaccine_etl

# In[21]:


vaccine_etl = covid_vaccine_data_without_india


# <font size=3>But when I closelly looked into the dataset then I found that on 20-06-2021 each column is empty. So I am <font color=red>dopping </font> this row

# In[22]:


vaccine_etl = vaccine_etl[vaccine_etl['Total_Doses_Administered'].notnull()]
vaccine_etl


# <font size=3>Checking that null value is replaced or not

# In[23]:


vaccine_etl.isnull().sum()


# We have successfully dropped null value

# <font size=4><font color=red>Outliers Handling

# <font size=4><font color=green>z-score or the interquartile range (IQR)</font>
# <br><font size=3>It is statistical technique.

# In[24]:


import numpy as np

# Calculate the IQR
Q1 = np.percentile(vaccine_etl['Total_Doses_Administered'], 25)
Q3 = np.percentile(vaccine_etl['Total_Doses_Administered'], 75)

# Ensure Q1 and Q3 are valid values
if Q1 is not None and Q3 is not None:
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = vaccine_etl[(vaccine_etl['Total_Doses_Administered'] < lower_bound) | (vaccine_etl['Total_Doses_Administered'] > upper_bound)]

    print(outliers)
else:
    print("Unable to calculate IQR due to missing values.")



# <font size=4><font color=green>Box Plot Method</font>
# <br><font size=3>1. xlabel --> column
# <br>2. ylabel --> values
# <br>3. title --> Box Plot of column_name

# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.boxplot(y=vaccine_etl.Total_Doses_Administered)
plt.xlabel('Total_Doses_Administered')
plt.ylabel('Values')
plt.title('Box Plot of Total_Doses_Administered')
plt.show()


# <font size=3>From <font color=red>Z-score</font> and<font color=red> Box Plot Method</font>, we have observed that data is randomly distributed and exponentially high in most of the cases. So, I concluded that we can't consider it as Outliers otherwise we'll lose most of the data.

# <font size=3>Ploting <font color=green>Pie chart</font> among these Male_(Doses_Administered), Female_(Doses_Administered), Transgender_(Doses_Administered)

# In[26]:


import plotly.express as px

male = vaccine_etl['Male_(Doses_Administered)'].sum()
female = vaccine_etl['Female_(Doses_Administered)'].sum()
trans = vaccine_etl['Transgender_(Doses_Administered)'].sum()

total = male + female + trans

male_percentage = (male / total) * 100
female_percentage = (female / total) * 100
trans_percentage = (trans / total) * 100

fig = px.pie(names=["Male", "Female", "Trans"], values=[male_percentage, female_percentage, trans_percentage], title="Male vs Female vs Transgender (Doses_Administered)")
fig.show()


# <font size=3>👆👆This Pie Chart showing the distribution of Doses_Administered among Male, Female and Transgender where
# <br>1. Male are vaccinated 53%
# <br>2. Female are vaccinated 46.7%
# <br>3. Transgender are vaccinated 0.02%

# <font size=3><font color=green>Visualisation of missing value</font> vaccine_etl['Male_(Doses_Administered)','Female_(Doses_Administered)','Transgender_(Doses_Administered)'] with respect to vaccine_date

# In[27]:


import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the dataset
train = vaccine_etl

# Step 3: Convert the 'vaccine_date' column to datetime
train['vaccine_date'] = pd.to_datetime(train['vaccine_date'])

# Step 4: Sort the dataframe based on the 'vaccine_date' column
train.sort_values('vaccine_date', inplace=True)

# Step 5: Display the missing value matrix for the specific column based on the date
msno.matrix(train[['vaccine_date', 'Male_(Doses_Administered)','Female_(Doses_Administered)','Transgender_(Doses_Administered)']])

# Step 6: Show the plot
plt.show()


# <font size=3>👆There is zero null value

# <font size=5><font color=green>===============================================================

# In[28]:


print("1. Total Individuals Vaccinated")
print("Null value %: ",format(1657/7416*100,'0.2f'))


# <font size=3><font color=green>Visualisation of missing value</font> vaccine_etl['Total_Individuals_Vaccinated']

# In[29]:


import missingno as msno
import pandas as pd

# Step 2: Load the dataset
train = vaccine_etl

# Step 3: Display the missing value matrix for the specific column
msno.matrix(train['Total_Individuals_Vaccinated'].to_frame())

# Step 4: Show the plot
plt.show()


# <font size=3><font color=green>Visualisation of missing value </font>vaccine_etl['Total_Individuals_Vaccinated'] with respect to vaccine_date

# In[30]:


import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the dataset
train = vaccine_etl

# Step 3: Convert the 'vaccine_date' column to datetime
train['vaccine_date'] = pd.to_datetime(train['vaccine_date'])

# Step 4: Sort the dataframe based on the 'vaccine_date' column
train.sort_values('vaccine_date', inplace=True)

# Step 5: Display the missing value matrix for the specific column based on the date
msno.matrix(train[['vaccine_date', 'Total_Individuals_Vaccinated']])

# Step 6: Show the plot
plt.show()


# <font size=3>We can observe that missing values on Total_Individuals_Vaccinated is continuous not distributed randomly so we can't use Mean or Median here. Becuase if we do so Total_Individuals_Vaccinated column will be biased.

# <font size=5><font color=gree>Top 5 Most Vaccinated State in India

# In[31]:


#Most vaccinated State
max_vac_state=vaccine_etl.groupby('State')['Total_Individuals_Vaccinated'].sum().to_frame('Total_Individuals_Vaccinated')
max_vac_state=max_vac_state.sort_values(by='Total_Individuals_Vaccinated',ascending=False)[:5]
max_vac_state


# <font size=3><font color=green>Ploting BarPlot of 'Top 5 Vaccinated States in India'

# In[32]:


fig=plt.figure(figsize=(12,5))
plt.title("Top 5 Vaccinated States in India",size=16)
x=sns.barplot(data=max_vac_state.iloc[:10],y=max_vac_state.Total_Individuals_Vaccinated,x=max_vac_state.index,linewidth=2,edgecolor='black')


# <font color=red><font size=4>Observation</font>
# <font size=3><font color=black><br>1. Maharashtra had most active cases and it had also more vaccinated people
# <br>2. Karnataka, Kerala, TamilNadu failed to vaccinate their people as per their active cases thats why after Maharashtra they had more deaths due to covid.
# <br>3. Maharastra and Uttar Pradesh did quit well to control covid cases.

# <font size=5><font color=red>Least 5 Vaccinated State in India

# In[33]:


#Most vaccinated State
min_vac_state=vaccine_etl.groupby('State')['Total_Individuals_Vaccinated'].sum().to_frame('Total_Individuals_Vaccinated')
min_vac_state=min_vac_state.sort_values(by='Total_Individuals_Vaccinated',ascending=True)[:5]
min_vac_state


# <font size=3><font color=green>Ploting BarPlot of 'Least 5 Vaccinated States in India'
# <font color=black><br>1. Take Figsize = 12,5 and title size = 16
# <br>2. Create Bar Plot with linewidth = 2 and edgecolor = black
# <br>3. Wrap x-axis Label text with rotation = 45 and ha = right

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

fig = plt.figure(figsize=(12, 5))
plt.title("Least 5 Vaccinated States in India", size=16)

# Create the bar plot
x = sns.barplot(
    data=min_vac_state.iloc[:10],
    y=min_vac_state.Total_Individuals_Vaccinated,
    x=min_vac_state.index,
    linewidth=2,
    edgecolor='black'
)

# Wrap x-axis label text
labels = [ '\n'.join(wrap(label.get_text(), 15)) for label in x.get_xticklabels() ]
x.set_xticklabels(labels, rotation=45, ha='right')

plt.tight_layout()
plt.show()


# We can observer that Lakshadeep is Least vaccinated state in India followed by Andaman and Nicobar Island

# <font size=5><font color=green>===============================================================

# In[35]:


print("1. AEFI")
print("Null value %: ",format(2125/7416*100,'0.2f'))


# <font size=3>Column AEFI is not required so I am droping this column 

# In[36]:


vaccine_etl = vaccine_etl.drop(columns=['AEFI'],axis=1)


# <font size=3>Successfully droped AEFI column

# <font size=5><font color=green>===============================================================

# In[37]:


print("1. 45-60 Years(Individuals Vaccinated)")
print("2. 60+ Years(Individuals Vaccinated)")
print("3. 18-44 Years(Individuals Vaccinated)")

print("Null value %: ",format(3783/7416*100,'0.2f'))


# <font size=3><font color=green>Visualisation of missing value vaccine_et['45-60_Years(Individuals_Vaccinated)','60+_Years(Individuals_Vaccinated)','18-44_Years(Individuals_Vaccinated)'] with respect to vaccine_date

# In[38]:


import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the dataset
train = vaccine_etl

# Step 3: Convert the 'vaccine_date' column to datetime
train['vaccine_date'] = pd.to_datetime(train['vaccine_date'])

# Step 4: Sort the dataframe based on the 'vaccine_date' column
train.sort_values('vaccine_date', inplace=True)

# Step 5: Display the missing value matrix for the specific column based on the date
msno.matrix(train[['vaccine_date', '45-60_Years(Individuals_Vaccinated)','60+_Years(Individuals_Vaccinated)','18-44_Years(Individuals_Vaccinated)']])

# Step 6: Show the plot
plt.show()


# <font size=3>We can observe that missing values on <font color=green>'45-60_Years(Individuals_Vaccinated)','60+_Years(Individuals_Vaccinated)','18-44_Years(Individuals_Vaccinated)' </font>is continuous not distributed randomly so we can't use Mean or Median here. Becuase if we do so '45-60_Years(Individuals_Vaccinated)','60+_Years(Individuals_Vaccinated)','18-44_Years(Individuals_Vaccinated)' column will be biased.

# <font size=5><font color=green>===============================================================

# In[39]:


print("1. Sputnik V (Doses Administered)")
print("Null value %: ",format(4502/7416*100,'0.2f'))


# <font size=3><font color=green>Visualisation of missing value vaccine_et['CoviShield_(Doses_Administered)','_Covaxin_(Doses_Administered)','Sputnik_V_(Doses_Administered)'] with respect to vaccine_date

# In[40]:


import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the dataset
train = vaccine_etl

# Step 3: Convert the 'vaccine_date' column to datetime
train['vaccine_date'] = pd.to_datetime(train['vaccine_date'])

# Step 4: Sort the dataframe based on the 'vaccine_date' column
train.sort_values('vaccine_date', inplace=True)

# Step 5: Display the missing value matrix for the specific column based on the date
msno.matrix(train[['vaccine_date', '_Covaxin_(Doses_Administered)','CoviShield_(Doses_Administered)','Sputnik_V_(Doses_Administered)']])

# Step 6: Show the plot
plt.show()


# <font size=3>Since we have only three vaccines and Sputnki_v(Doses_Administered) has more number of missing values but i am keeping this so that we know when this vaccine came to market and acquired it.

# <font size=3>Ploting <font color=green>Pie chart</font> among these _Covaxin_(Doses_Administered), CoviShield_(Doses_Administered), Transgender_(Doses_Administered)

# In[41]:


import plotly.express as px

covaxin = vaccine_etl['_Covaxin_(Doses_Administered)'].sum()
covishield = vaccine_etl['CoviShield_(Doses_Administered)'].sum()
sputnik = vaccine_etl['Sputnik_V_(Doses_Administered)'].sum()

total = covaxin + covishield + sputnik

covaxin_percentage = (covaxin / total) * 100
covishield_percentage = (covishield / total) * 100
sputnik_percentage = (sputnik / total) * 100

fig = px.pie(names=["covaxin", "covishield", "sputnik V"], values=[covaxin_percentage, covishield_percentage, sputnik_percentage], title="covaxin vs covishield vs sputnik_V (Doses_Administered)")
fig.show()


# <font size=3>👆👆We can observe that 
# <br>1. Covishield is most used vaccine in India with 88.6%.
# <br>2. Sputnik V is least used vaccine in India with just 0.041%

# <font size=5><font color=green>===============================================================

# In[42]:


print("1. 45-60 Years (Doses Administered)")
print("2. 60+ Years (Doses Administered)")
print("3. 18-44 Years (Doses Administered)")
print("Null value %: ",format(5760/7416*100,'0.2f'))


# <font size=3>Since Null value is more than 77% which is very high and if i try to fill these null values with Mean, Median or Mode then data will be biased so I am <font color=red>dropping </font> these columns

# In[43]:


vaccine_etl=vaccine_etl.drop(columns=['18-44_Years_(Doses_Administered)','45-60_Years_(Doses_Administered)','60+_Years_(Doses_Administered)'],axis=1)


# <font size=3>We have successfully droped these columns.

# <font size=5><font color=green>===============================================================

# In[44]:


print("1. Transgender(Individuals Vaccinated)")
print("2. Male(Individuals Vaccinated)")
print("3. Female(Individuals Vaccinated)")
print("Null value %: ",format(7416/7416*100,'0.2f'))


# <font size=3>Since null value is 100% so there is atall no use of these columns so i am <font color=red>droping</font> it.

# In[45]:


vaccine_etl=vaccine_etl.drop(columns=['Male(Individuals_Vaccinated)','Female(Individuals_Vaccinated)','Transgender(Individuals_Vaccinated)'],axis=1)


# <font size=3>We have succefully <font color=green>droped </font> these coloumn.

# <font size=4><font color=red>Calculation of Loss of Data

# In[46]:


print('Initial_dataset:', covid_vaccine_data.shape)
print('Cleaned_dataset: ',vaccine_etl.shape)


# In[47]:


loss_of_columns = 7845 - 7415
percentage_loss = (loss_of_columns / 7845) * 100
print("Loss of columns:", loss_of_columns, "columns, or", format(percentage_loss, '.2f'), "%")

loss_of_rows = 24 - 17
percentage_loss_rows = (loss_of_rows / 24) * 100
print("Loss of rows:", loss_of_rows, "rows, or", format(percentage_loss_rows, '.2f'), "%")


# <font size=5><font color=red>Saving/Downloading vaccine_etl as csv file

# In[48]:


vaccine_etl.to_csv('vaccine_etl.csv',index=False)


# <font size=3>File is downloaded successfully

# In[ ]:




