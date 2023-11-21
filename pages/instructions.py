import streamlit as st





st.title("Instructions :")
st.subheader("Upload Your Data")
st.markdown("1. Click the 'Choose a CSV file' button and select the CSV file containing your data.")
st.markdown("2. Ensure that the CSV file has a header row and that the target variable (the variable you want to predict) is located in the last column.")
st.markdown("3. Click the 'Import Data' button to import the data and proceed with the analysis.")
st.subheader("Enter Your Input Values")
st.markdown("1. For each feature (column) in the dataset, enter the corresponding value in the respective text input field.")
st.markdown("2. Ensure that the input values match the data type expected for each feature. For instance, if a feature represents a numerical value, enter a numeric value.")
st.markdown("3. Click the 'Predict' button to generate predictions based on your input values.")
st.subheader("Interpret the Results")
st.markdown("1. Observe the predictions for each model displayed in the 'Predictions' table.")
st.markdown("2. Analyze the accuracies of each model presented in the 'Accuracies' table.")
st.markdown("3. Identify the top 3 models based on their accuracies and review their predictions and accuracies.")
st.markdown("4. Examine the comparison graph to visualize the relative performance of the different models.")