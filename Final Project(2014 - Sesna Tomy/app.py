import os
import cv2
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import streamlit as st
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input



st.set_page_config(
    page_title="Image Classification App",
    page_icon="✨", 
    initial_sidebar_state="expanded",
)

#layout="wide",


class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
IMAGE_SIZE = (150, 150)


def load_test_data():
    """
    Load the test data:
        - 3,000 images to evaluate how accurately the network learned to classify images.
    """
    test_dataset = '/home/sesna/sesna/image_classification/seg_test'
    test_images = []
    test_labels = []

    print("Loading test data from {}".format(test_dataset))

    # Iterate through each folder corresponding to a category
    for folder in os.listdir(test_dataset):
        label = class_names_label[folder]

        # Iterate through each image in our folder
        for file in tqdm(os.listdir(os.path.join(test_dataset, folder))):
            # Get the path name of the image
            img_path = os.path.join(os.path.join(test_dataset, folder), file)

            # Open and resize the img
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)

            # Append the image and its corresponding label to the output
            test_images.append(image)
            test_labels.append(label)

    test_images = np.array(test_images, dtype='float32')
    test_labels = np.array(test_labels, dtype='int32')

    return test_images, test_labels





def plot_accuracy_loss(history):
            """
            Plot the accuracy and the loss during the training of the nn.
            """
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

            # Plot accuracy
            axes[0].plot(history['accuracy'], 'bo--', label="acc")
            axes[0].plot(history['val_accuracy'], 'ro--', label="val_acc")
            axes[0].set_title("Train Accuracy vs Validation Accuracy")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_xlabel("Epochs")
            axes[0].legend()

            # Plot loss function
            axes[1].plot(history['loss'], 'bo--', label="loss")
            axes[1].plot(history['val_loss'], 'ro--', label="val_loss")
            axes[1].set_title("Train Loss vs Validation Loss")
            axes[1].set_ylabel("Loss")
            axes[1].set_xlabel("Epochs")

            axes[1].legend()

            # Display the plots in Streamlit
            st.pyplot(fig)



def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plot a confusion matrix and display it in a Streamlit app.
    """
    CM = confusion_matrix(true_labels, predicted_labels)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots()
    
    # Create a heatmap using Seaborn
    sns.heatmap(CM, annot=True, annot_kws={"size": 10}, 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    # Set the title
    ax.set_title('Confusion Matrix')
    
    # Display the plot in Streamlit
    st.pyplot(fig)



model1 = load_model('/home/sesna/sesna/image_classification/models/imageclassifier1.h5')

tab1, tab2, tab4 = st.tabs(["Classifier", "About Dataset","Models"])

with tab1:
    st.title("Image Classification")
    st.subheader("Input")

    uploaded_file = st.file_uploader(
       "Choose an image to classify", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        # Read image as bytes
        img_bytes = uploaded_file.read()

        # Convert bytes to a NumPy array
        img_np = np.frombuffer(img_bytes, np.uint8)

        # Decode the image using OpenCV (cv2)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)


        # Resize the image using OpenCV (cv2)
        new_size = (150,150)  # Set your desired size
        img_resized = cv2.resize(img, new_size)


        # Normalize and expand dimensions for model prediction
        scaled_img = np.expand_dims(img_resized / 255, 0)

        # Load your Keras model
        model1 = load_model('/home/sesna/sesna/image_classification/models/imageclassifier1.h5')

        # Make predictions
        pred = model1.predict(scaled_img)
        st.write(pred)
        # st.write(pred)
        maxi = max(pred[0])
        st.write(maxi)
    
        max_index = np.argmax(pred)
        max_class_name = [k for k, v in class_names_label.items() if v == max_index][0]

        st.write(f"The picture is of a {max_class_name}")
        # Display the image using Streamlit
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image",width=500)


with tab2:
    IMAGE_SIZE = (150, 150)

    def calculate_total_samples(dataset_paths):
        total_samples = 0
        count_class = {}
        for dataset in dataset_paths:
            for folder in os.listdir(dataset):
            # Assuming each folder contains the same number of images
                num_images_in_folder = len(os.listdir(os.path.join(dataset, folder)))
                count_class[folder] = num_images_in_folder
                total_samples += num_images_in_folder

        return total_samples, count_class

    train_data= ['/home/sesna/sesna/image_classification/data']
    test_data = ['/home/sesna/sesna/image_classification/seg_test']
    total_samples, count_of_class = calculate_total_samples(train_data)
    total_train ,count_of_class = calculate_total_samples(test_data)
    each_class = pd.DataFrame.from_dict(count_of_class, orient='index', columns=['Count'])
    st.write("Total Training data :", total_samples)
    st.write("Total Testing data:", total_train)
    st.write("Total number of class labels:", 6)
    st.dataframe(each_class)
    st.write("Size of each image:", IMAGE_SIZE)
    st.bar_chart(each_class)

    
    df_count = pd.DataFrame(list(count_of_class.items()), columns=['Class', 'Count'])

    # Calculate percentage for each class
    df_count['Percentage'] = (df_count['Count'] / df_count['Count'].sum()) * 100

#    Display the percentage pie chart using Streamlit and Plotly
    fig = px.pie(df_count, values='Percentage', names='Class', title='Proportion of each observed category')
    st.plotly_chart(fig)




with tab4:
    st.markdown("## Simple Model")
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)
        
    # Assuming you have the 'history' variable from your model training
    # You can call the function like this in your Streamlit app
    plot_accuracy_loss(history)
    
    # Now, you can call the function to get the test data
    test_images, test_labels = load_test_data()

    model = load_model('/home/sesna/sesna/image_classification/models/imageclassifier.h5')
    predictions = model.predict(test_images)     # Vector of probabilities
    pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
    plot_confusion_matrix(test_labels,pred_labels,class_names)
    accuracy = accuracy_score(test_labels, pred_labels)
    st.write(f"Accuracy of the simple model:",{accuracy})

    st.markdown("## Model After Transfer Learning")
    with open('training_history1.pkl', 'rb') as file:
        history1 = pickle.load(file)


    plot_accuracy_loss(history1)


    prediction = model1.predict(test_images)     # Vector of probabilities
    pred_label = np.argmax(prediction, axis = 1)
    plot_confusion_matrix(test_labels,pred_label,class_names)

    accuracy1 = accuracy_score(test_labels, pred_label)
    st.write(f"Accuracy of the model after Transfer Leatning:",{accuracy1})

    st.write("The classifier has trouble with 2 kinds of images.")
    st.write("It has trouble with street and buildings. Well, it can be understandable as there are buildings in the street.")
    st.write("It has also trouble with sea, glacier and mountain as well. However, it can detects forest very accurately!")









