#Import necessary Python libraries for building the Streamlit app, data manipulation, visualization, and machine learning. 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Loading the data and removing the duplicate rows.
df = pd.read_csv("/home/sesna/stremlit/creditcard.csv")
df.drop_duplicates(inplace=True)

#Selecting data for prediction
samples_0= df[df['Class'] == 0].sample(n=20, random_state=42)
samples_1 = df[df['Class'] == 1].sample(n=20, random_state=42)
sample_0 = samples_0.drop('Class',axis=1)
sample_1 = samples_1.drop('Class',axis=1)
df2= pd.concat([sample_1, sample_0])
df = df.drop(df2.index).reset_index(drop=True)
sc = StandardScaler()
df2['Amount']=sc.fit_transform(pd.DataFrame(df2['Amount']))

#Set Streamlit page configuration, title, and display dataset information using Streamlit functions.
st.set_page_config(page_title= "Project 1")
st.title("Credit Card Fraud Detection Model")
st.write("Dataset")
st.dataframe(df)
st.write("Shape of Dataframe:",df.shape)

#Display descriptive statistics of the dataset if the user checks the checkbox in the Streamlit sidebar.
if st.sidebar.checkbox("Show description"):
    st.write("Data description:\n")
    description = df.describe()
    st.dataframe(description)

#Create a count plot to visualize the distribution of classes (fraudulent and non-fraudulent transactions).
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
ax.set_title('Distribution of class')
ax.set_ylabel('Number of occurance')
ax.set_xlabel('Class')
ax.grid(True, linestyle='--', color='black', linewidth=0.5)
st.pyplot(fig)

#Provide a written observation about the class distribution in the dataset.
value = len(df[df['Class'] == 1])
value1 = len(df[df['Class']==0])
percentage = value/len(df) * 100
st.write(f"Observation: The data is unbalanced with {value} fraud transactions and {value1} normal transactions. Tthe positive class (frauds) account for {percentage} of all transactions.")

#Create histograms to visualize the distribution of legitimate and fraudulent transactions over time and based on transaction amount.
fig, ax = plt.subplots()
ax.hist(df.Time[df.Class == 0],bins=45)
ax.set_title('Legitimate Transactions per Second')
ax.set_ylabel('Transactions')
ax.set_xlabel('Time (Seconds)')
ax.grid(True, linestyle='--', color='black', linewidth=0.5)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist(df.Amount[df.Class == 0],bins = 5)
ax.set_title('Legitimate Transactions Based on Amount')
ax.set_ylabel('Transactions')
ax.set_xlabel('Transaction Amount')
ax.grid(True, linestyle='--', color='black', linewidth=0.5)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist(df.Time[df.Class == 1],bins = 45)
ax.set_title('Fraudulent Transactions per Second')
ax.set_ylabel('Transactions')
ax.set_xlabel('Time (Seconds)')
ax.grid(True, linestyle='--', color='black', linewidth=0.5)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist(df.Amount[df.Class == 1])
ax.set_title('Fraudulent Transactions Based on Amount')
ax.set_ylabel('Transactions')
ax.set_xlabel('Transaction Amount')
ax.grid(True, linestyle='--', color='black', linewidth=0.5)
st.pyplot(fig)
st.write("The first and third graphs show that time has no effect on the transactions")
st.write("The second and fourth graphs shows that the value of the common transactions stays between 0 and 5000,while the fraudulent transactions has very few value above 500.")

#Create a balanced sample for modeling by randomly sampling normal transactions and Scale the 'Amount' column.
normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]
normal_sample = normal.sample(n = len(fraud),random_state=42)
df1 = pd.concat([normal_sample, fraud], ignore_index=True)
df1['Amount']=sc.fit_transform(pd.DataFrame(df1['Amount']))
df1.drop(['Time'],axis=1,inplace=True)

#Split the data into training and testing sets for machine learning modeling.
x = df1.drop('Class',axis = 1)
y = df1['Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)  

#Define feature names and initialize a list (selected_data) to store user-selected feature values for prediction.
feature_names = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount',]
selected_data = [0] * 29

#Define a function (make_prediction) to allow users to select features and make predictions using a logistic regression model.
def make_prediction():
    st.sidebar.title("Select Features")
    model = LogisticRegression()
    model.fit(x_train,y_train)

    for i, feature_name in enumerate(feature_names):
      
      selected_data[i] = st.sidebar.selectbox(f"Enter {feature_name}", df2[feature_name])

    if st.sidebar.button("Predict"):
        input_data = np.array(selected_data).reshape(1, -1)  
        result = model.predict(input_data)

        if result == 0:
            st.sidebar.write("This is a Normal transaction ")
        else:
            st.sidebar.write("This is a Fraud transaction ")

#Define a function (interactive_ploting) to create an interactive scatter plot based on user-selected axes.
def interactive_ploting():
    df2 = df.drop(['Time','Class'],axis=1)
    x_axis = st.sidebar.selectbox("Select X-axis",options=df2.columns)
    y_axis = st.sidebar.selectbox("Select y-axis",options=df2.columns)
    fig, ax = plt.subplots()
    ax.scatter(df2[x_axis],df2[y_axis])
    ax.set_xlabel(f"{x_axis}")
    ax.set_ylabel(f"{y_axis}")
    ax.grid(True, linestyle='--', color='black', linewidth=1)
    st.write(f"Scatter plot: {x_axis} vs {y_axis}")
    st.pyplot(fig)

#Execute the prediction and interactive plotting functions when the script is run as the main program.
if __name__ == '__main__':
    make_prediction()
    interactive_ploting()






















