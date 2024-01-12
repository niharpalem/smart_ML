import streamlit as st
import pandas as pd
import os
import sweetviz as sv

# Deferred imports for pycaret
from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*qp2mD4aDTWCQFYDrAPN8uQ@2x.jpeg")
    st.title("Auto ML")
    choice = st.radio("Navigation", ['Upload', 'Profiling', 'ML', 'Download'])
    st.info('Application for automated ML pipeline using Streamlit')
st.write('Smart ML')

# Initialize df to None
df = None

# Check if the dataset exists and load it
if os.path.exists("source_data.csv"):
    df = pd.read_csv('source_data.csv', index_col=None)

if choice == 'Upload':
    st.title("Upload your dataset")
    file = st.file_uploader('dataset')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('source_data.csv', index=None)
        st.dataframe(df)

if choice == 'Profiling' and df is not None:
    st.title("Profile EDA")
    report = sv.analyze(df)
    report.show_html()  # This will generate and open the HTML report

if choice == 'ML' and df is not None:
    st.title("Machine Learning")
    target = st.selectbox("Select the target Variable", df.columns)
    if st.button("Train Model"):
        setup(df, target=target, silent=True)
        setupdf = pull()
        st.info("experiment settings")
        st.dataframe(setupdf)
        best_model = compare_models()
        comparedf = pull()
        st.info("ML model")
        st.dataframe(comparedf)
        st.write(best_model)
        save_model(best_model, 'best_model.pkl')

if choice == 'Download':
    # Check if the model file exists before creating the download button
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download the model", f, 'best_model.pkl')
    else:
        st.error("No model to download. Please train a model first.")