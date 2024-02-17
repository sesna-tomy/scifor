import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model



st.set_page_config(
    page_title="Image Classification App",
    page_icon="✨",     
    layout='wide',                                          
    initial_sidebar_state="expanded",
)

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
IMAGE_SIZE = (150, 150)



def plot_accuracy_loss(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axes[0].plot(history['accuracy'], 'bo--', label="acc")            # Plot accuracy
    axes[0].plot(history['val_accuracy'], 'ro--', label="val_acc")
    axes[0].set_title("Train Accuracy vs Validation Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()

    axes[1].plot(history['loss'], 'bo--', label="loss")              # Plot loss function
    axes[1].plot(history['val_loss'], 'ro--', label="val_loss")
    axes[1].set_title("Train Loss vs Validation Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()

    st.pyplot(fig)                                                  # Display the plots in Streamlit




def calculate_total_samples(dataset_paths):
    total_samples = 0
    count_class = {}
    for dataset in dataset_paths:
        for folder in os.listdir(dataset):
            num_images_in_folder = len(os.listdir(os.path.join(dataset, folder)))     # Assuming each folder contains the same number of images
            count_class[folder] = num_images_in_folder
            total_samples += num_images_in_folder

    return total_samples, count_class

def disply_image(folder_path):
    image_files = os.listdir(folder_path)

    for i in range(0,5):
    # Construct the full path to the image
        image_path = os.path.join(folder_path, image_files[i])
        st.image(image_path,caption=image_files[i],width=200)






model1 = load_model('/home/sesna/sesna/image_classification/models/imageclassifier1.h5')
tab1, tab2, tab3 , tab4 = st.tabs(["Classifier", "About Dataset","Models","Training data"])


with tab1:
    st.title("Image Classification")
    st.subheader("Input")

    uploaded_file = st.file_uploader(
       "Choose an image to classify", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img_bytes = uploaded_file.read()                                                               # Read image as bytes
        img_np = np.frombuffer(img_bytes, np.uint8)                                                    # Convert bytes to a NumPy array
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)                                                   # Decode the image using OpenCV (cv2)
        new_size = (150,150)                                                                           # Resize the image using OpenCV (cv2) Set your desired size
        img_resized = cv2.resize(img, new_size)
        scaled_img = np.expand_dims(img_resized / 255, 0)                                              # Normalize and expand dimensions for model prediction
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image",width=500)             # Display the image using Streamlit
        
        if st.button('predict'):
            model1 = load_model('/home/sesna/sesna/image_classification/models/imageclassifier1.h5')   # Load your Keras model
            pred = model1.predict(scaled_img)                                                          # Make predictions
            each_class = pd.DataFrame(pred, columns=class_names)
            st.write("Probability of each class")
            st.dataframe(each_class)                                                                   # st.write(pred)
            maxi = max(pred[0])
    
            max_index = np.argmax(pred)
            max_class_name = [k for k, v in class_names_label.items() if v == max_index][0]
            st.write(f'{max_class_name} has the higest probability of {maxi:.3f}')
            st.write(f"The picture is of a {max_class_name}")
        
        


with tab2:
    IMAGE_SIZE = (150, 150)
    col1, col2 = st.columns(2)
    train_data= ['/home/sesna/sesna/image_classification/data']
    test_data = ['/home/sesna/sesna/image_classification/seg_test']
    total_samples, count_of_class1 = calculate_total_samples(train_data)
    total_train ,count_of_class = calculate_total_samples(test_data)
    each_class = pd.DataFrame.from_dict(count_of_class, orient='index', columns=['Count'])
    each_class1 = pd.DataFrame.from_dict(count_of_class1, orient='index', columns=['Count'])
    st.write("Total Training data :", total_samples)
    st.write("Total Testing data:", total_train)
    st.write("Total number of class labels:", 6)
    with col1:
        st.write('Training data')
        st.dataframe(each_class1)
    with col2:
        st.write('Testing data')
        st.dataframe(each_class)
    st.write("Size of each image:", IMAGE_SIZE)
    st.bar_chart(each_class1)

    
    df_count = pd.DataFrame(list(count_of_class1.items()), columns=['Class', 'Count'])                          # Calculate percentage for each class
    df_count['Percentage'] = (df_count['Count'] / df_count['Count'].sum()) * 100
    fig = px.pie(df_count, values='Percentage', names='Class', title='Proportion of each observed category')    # Display the percentage pie chart using Streamlit and Plotly
    st.plotly_chart(fig)




with tab3:
    st.markdown("## Simple Model")
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)

    plot_accuracy_loss(history)
    st.image('/home/sesna/sesna/image_classification/confusion.png',width=500)
                   
    st.write(f"Accuracy of the simple model is 66.86%")

    st.markdown("## Model After Transfer Learning")
    with open('training_history1.pkl', 'rb') as file:
        history1 = pickle.load(file)

    plot_accuracy_loss(history1)
    st.image('/home/sesna/sesna/image_classification/confusion1.png',width=500)

    st.write(f"Accuracy of the model after Transfer Learning is 85.67%",)

    st.write("The classifier has trouble with 2 kinds of images.")
    st.write("It has trouble with street and buildings. Well, it can be understandable as there are buildings in the street.")
    st.write("It has also trouble with sea, glacier and mountain as well. However, it can detects forest very accurately!")


with tab4:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.write(class_names[0])
        disply_image('/home/sesna/sesna/image_classification/data/mountain')

    with col2:
        st.write(class_names[1])
        disply_image('/home/sesna/sesna/image_classification/data/street')

    with col3:
        st.write(class_names[2])
        disply_image('/home/sesna/sesna/image_classification/data/glacier')

    with col4:
        st.write(class_names[3])
        disply_image('/home/sesna/sesna/image_classification/data/buildings')

    with col5:
        st.write(class_names[4])
        disply_image('/home/sesna/sesna/image_classification/data/sea')

    with col6:
        st.write(class_names[5])
        disply_image('/home/sesna/sesna/image_classification/data/forest')







