# <span style="font-size:32px;">**Credit Card Fraud Detection Model**</span>

## <span style="font-size:24px;">**Overview**</span>
This project implements a Credit Card Fraud Detection Model using Python, Streamlit, and machine learning techniques. 
The aim is to detect fraudulent transactions in a credit card dataset and provide visualizations for better understanding.

## <span style="font-size:24px;">**Features**</span>
**Data Exploration:** The project provides a comprehensive exploration of the dataset, including visualizations of class distribution, time-based transactions, and amount distribution.

**Data Preprocessing:** The dataset is preprocessed to handle duplicates, scale the 'Amount' column, and balance the class distribution.

**Machine Learning Model:** Logistic Regression is used to build a fraud detection model. Users can input features through the Streamlit interface, and the model predicts whether a transaction is normal or fraudulent.

**Interactive Plotting:** The application allows users to create scatter plots based on selected features to gain insights into the data distribution.

## <span style="font-size:24px;">**Requirements**</span>
    Python 3.x:   
              Python is a high-level, general-purpose programming language known for its readability and versatility. Version 3.x is the latest major release with improvements and new features.  
    Streamlit:  
              Streamlit is a Python library for creating web applications with minimal code. It simplifies the process of turning data scripts into shareable web apps.
    pandas:  
              Pandas is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrame for efficient data handling and manipulation.
    numpy:  
              NumPy is a numerical computing library in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on them.
    matplotlib:  
              Matplotlib is a 2D plotting library for Python. It enables the creation of static, animated, and interactive visualizations in Python.
    seaborn:  
              Seaborn is a statistical data visualization library based on Matplotlib. It provides a high-level interface for creating informative and attractive statistical graphics.
    scikit-learn:  
              Scikit-learn is a machine learning library for Python. It offers simple and efficient tools for data analysis and modeling, including various algorithms for classification, regression, clustering, and more.
## <span style="font-size:24px;">**Setup**</span>
**Clone the repository:**
  git clone https://github.com/sesna-tomy/scifor.git

**Changing the directory:**
  cd Project_1

**Install the required dependencies:**
  pip install -r requirements.txt

**Run the Streamlit app:**
  streamlit run credit_card.py

## <span style="font-size:24px;">**Usage**</span>
**1.** Open the Streamlit app in your browser (by default, it runs on localhost:8501).

**2.** Explore the dataset and visualizations.

**3.** Use the sidebar to select features and make predictions with the built-in fraud detection model.

**4.** Create scatter plots interactively to analyze data distribution.


