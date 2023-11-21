import streamlit as st
import pydeck as pdk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from io import BytesIO
import PIL.Image
# import pycaret
# from pycaret.regression import *
# from pycaret.datasets import get_data


st.title("Predictor App")


uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False , type=['csv'] )


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    feature_names = list(df.columns)[:-1]
    X = df.iloc[:, :-1]
    X.fillna(0, inplace=True)
    y = df.iloc[:, -1]
    y.fillna(0, inplace=True)


    

    input_values = []
    for feature_name in list(df.columns)[:-1]:
        input_value = st.text_input(f'{feature_name}', placeholder="Enter...")

        # Check if the input value is a string
        if isinstance(input_value, str):
            # Try converting the input value to a numeric value
            try:
                input_value = float(input_value)
            except:
                # Input value is not a valid numeric value, display an error message
                st.error(f"Invalid input for {feature_name}: '{input_value}'")
                continue  # Skip to the next iteration of the loop
        else:
            pass

        input_values.append(input_value)

    # If all input values are valid, continue with the prediction
    if len(input_values) == len(list(df.columns)[:-1]):
        # Make predictions using all models
        predictions = {
            "Linear Regression": None,
            "Random Forest": None,
            "Decision Tree": None,
            "Logistic Regression": None,
            "Support Vector Machine": None,
            "Naive Bayes": None
        }

        accuracies = {
            "Linear Regression": None,
            "Random Forest": None,
            "Decision Tree": None,
            "Logistic Regression": None,
            "Support Vector Machine": None,
            "Naive Bayes": None
        }

        for model_name, model in zip(["Linear Regression", "Random Forest", "Decision Tree", "Logistic Regression",
                                     "Support Vector Machine", "Naive Bayes"],
                                   [LinearRegression(), RandomForestClassifier(n_estimators=100, random_state=42),
                                    DecisionTreeClassifier(max_depth=5, random_state=42),
                                    LogisticRegression(random_state=42),
                                    SVC(kernel='rbf', random_state=42),
                                    GaussianNB()]):
            model.fit(X, y)
            predictions[model_name] = model.predict([input_values])
            accuracies[model_name] = model.score(X, y)

        for model_name, predictions_arr in predictions.items():
            if not np.issubdtype(predictions_arr.dtype, np.integer):
                predictions[model_name] = predictions_arr.astype(int)
        if st.button('Predict'):
           st.subheader("Predictions")
           predictions_df = pd.DataFrame(predictions, index=["Prediction"])
           st.table(predictions_df)
           st.subheader("Accuracies")
           accuracies_df = pd.DataFrame(accuracies, index=["Accuracy"])
           st.table(accuracies_df)



        # Sort the models based on accuracy
           sorted_models = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)

        # Display predictions and accuracies for the top 3 models
           st.header("Top 3 Models and Their Predictions")
           for i in range(3):
               model_name, accuracy = sorted_models[i]
               prediction = predictions[model_name]
               st.markdown(f"**{i+1}. {model_name} Accuracy: {accuracy:.2f} Prediction: {prediction}**")
        



        # Prepare data for the graph
           model_names = list(accuracies.keys())
           accuracies_list = list(accuracies.values())

        # Create the graph
           plt.figure(figsize=(10, 6))
           plt.bar(model_names, accuracies_list, color=['gold'])
           plt.xlabel('Model Names')
           plt.ylabel('Accuracy')  # Set y-axis label to 'Accuracy'
           plt.title('Comparison of Model Accuracies')
           plt.xticks(rotation=45)
           plt.tight_layout()

        # Convert the figure to a PIL.Image object
           buf = BytesIO()
           plt.savefig(buf, format='png')
           image = PIL.Image.open(buf)

        # Display the graph
           st.image(image, use_column_width=True, caption="Comparison of Model Accuracies")

        

       
        # predictions_list = list(predictions.values())

        # # Create the prediction graph
        # plt.figure(figsize=(10, 6))
        # plt.bar(model_names, predictions_list, color=['b', 'g', 'r', 'c', 'm', 'y'])
        # plt.xlabel('Model Names')
        # plt.ylabel('Predicted Value')  # Set y-axis label to 'Predicted Value'
        # plt.title('Comparison of Model Predictions')
        # plt.xticks(rotation=45)
        # plt.tight_layout()

        # # Convert the figure to a PIL.Image object
        # buf = BytesIO()
        # plt.savefig(buf, format='png')
        # image = PIL.Image.open(buf)

        # # Display the prediction graph
        # st.image(image, use_column_width=True, caption="Comparison of Model Predictions")
        # Plot bar graph to compare predictions




        # model_names = list(predictions.keys())
        # predicted_values = list(predictions.values())

        # plt.figure(figsize=(10, 6))
        # plt.bar(model_names, predicted_values)
        # plt.title("Comparison of Predictions by Different Models")
        # plt.xlabel("Model Name")
        # plt.ylabel("Predicted Value")
        # plt.show()


        



















        