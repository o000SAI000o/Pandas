"""
✅ Core Data Structures: Series, DataFrame
✅ Data Loading: CSV, SQL, Excel
✅ Data Cleaning: Handling missing values, duplicates
✅ Feature Engineering: New columns, binning
✅ Filtering & Selection: Boolean indexing
✅ Grouping & Aggregation: groupby(), pivot tables
✅ Merging Data: Combining multiple datasets
✅ Time Series Analysis: Working with dates
✅ Dashboarding: Matplotlib, Plotly
"""
#DataFrame & Series (The backbone of Pandas)
"""
1.Series: One-dimensional labeled array
DataFrame: Two-dimensional table
Everything in Pandas is built around these structures.
DataFrames are used in almost all ML workflows for preprocessing data.
"""
import pandas as pd

#creating a series
s = pd.Series([10,20,30] ,index = ['a','b','c'])
print(s)

#creating dataframe
data = {'name':['alice','bob','charlie'], 'age':[25,30,40]}
df = pd.DataFrame(data)
print(df)

"""2. Data Loading & Exporting
✅ Read/Write Data (CSV, Excel, SQL, JSON)
📌 Why Important?
ML and analytics depend on structured data sources (Databases, CSVs, APIs)."""

#read csv
df = pd.read_csv("data.csv")

#read Excel
df = pd.read_excel("data.xlsx")

#read JSON
df = pd.read_json("data.json")

#export
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index = False)

""" Data Exploration (EDA - Exploratory Data Analysis)
📌 Why Important?
Helps understand the data distribution before applying ML models."""
df.info()  #summary of dataframe(column names,non-null counts,types)
df.describe() #summary statistics
df.head(5)  #first 5 rows
df.tail(5)  #last five rows
df.shape    #rows ans columns count
df.columns  #column names
df.nunique() #unique values count per column
df.isnull().sum() #count missing values

"""4. Data Cleaning & Preprocessing
📌 Why Important?
Garbage in = Garbage out (ML models fail with dirty data).
Essential before feeding data into ML models."""

#✅ Handling Missing Data

#remove missing values
df.dropna(inplace=True)

#fill missing values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())  
# Fill missing with mean

#✅ Handling Duplicates
df.drop_duplicates(inplace=True)

#✅ Data Type Conversion
df['Age'] = df['Age'].astype(int) #convert to integer

"""5. Feature Engineering
📌 Why Important?
Creating better features improves ML model accuracy."""

#✅ Creating New Columns
# Ensure 'First_Name' and 'Last_Name' columns exist
if 'FirstName' in df.columns and 'LastName' in df.columns:
    df['Full_Name'] = df['FirstName'] + " " + df['LastName']
else:
    print("⚠️ Warning: 'First_Name' and 'Last_Name' columns are missing!")

if 'Age' in df.columns:
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 18, 36, 80], labels=['Teen', 'Adult', 'Senior'], include_lowest=True)
else:
    print("⚠️ Warning: 'age' column is missing!")

print("✅ Data processing complete!")
print(df.columns.to_list())

"""6. Filtering & Selecting Data
Needed for data slicing and feature selection in ML."""
#✅ Selecting Columns
df[['LastName','Age']]

#✅ Filtering Rows (Conditional Selection)
filtered_df = df[df['Age'] > 25]['FirstName']  #select rows where age>25
print(filtered_df)
#✅ Boolean Masking
masked_df =df[(df['Age'] > 25) & (df['City'] == 'New York')]
print(masked_df)

"""7. Grouping & Aggregation
📌 Why Important?

Used for summarizing data, essential in analytics & dashboards."""
#✅ Grouping Data
avg_df =df.groupby('City')['Salary'].mean() #avg salary per city
print(avg_df, end="")
#✅ Pivot Table (For Dashboards)
pivot_df=df.pivot_table(index='City', values='Salary', aggfunc='sum')
print(pivot_df,end="")
# summarize and aggregate data in a structured way.

"""8. Merging & Joining Datasets
📌 Why Important?
Real-world data is never in a single table. You must merge multiple datasets"""

#✅ Merge Two DataFrames
df1 = pd.DataFrame({'Age':[30,27],'Name':['Bob', 'eve']})
df2 = pd.DataFrame({'Age': [30,27], 'Salary': [60000, 70000]})
merged_df = pd.merge(df1, df2, on="Age")
print(merged_df, end=" ")

"""9. Time Series Analysis
📌 Why Important?
Used for stock market prediction, sales forecasting, etc."""
print(df.columns.to_list)
#✅ Convert to DateTime
#df['Date'] = pd.to_datetime(df['Date'])
#✅ Extract Time Components
"""
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
"""
"""10. Using Pandas for Dashboards
📌 How is Pandas Used in Dashboards?
Pandas helps process and prepare data for visualization tools like Matplotlib, Seaborn, and Plotly.
✅ Example:"""
#✅ Example: Pandas + Matplotlib Dashboard

import pandas as pd
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("sales_data.csv")
print(df.columns)
#process data
monthly_sales = df.groupby('Month')['Revenue'].sum()

#create a simple dashboard
plt.figure(figsize=(10,10))
#Creates a plotting area with a size of 10x10 inches.
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-')
#monthly_sales.index → Represents the months (Jan, Feb, Mar, ...).
#monthly_sales.values → Represents the total revenue for each month (5000, 7000, 8000, ...
#marker='o' → Adds circular points on the graph for each data point.
#linestyle='-' → Connects the points with a line.
plt.title("Monthly Sales Revenue")
# Title of the chart
plt.xlabel("Month")
# X-axis label
plt.ylabel("Revenue")
# y-axis label
plt.grid(True)
#Adds grid lines to make the chart easier to read
plt.show()
#Displays the final chart! 📊