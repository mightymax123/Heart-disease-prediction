import streamlit as st
import pandas as pd
import numpy as np
from utils import clean_df, get_preds, load_data, OneHotEncodeDf

data_loaded = load_data('app/data.pkl') ## this contains a dict w the trained model and the one hot encoding configs

def main(): ## could break this up into subfunctions
    st.title('ML app for heart disease detection')
    uploaded_file = st.file_uploader("Please choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df, dropped_idx = clean_df(df) ## removes cols with NA vals or where the oldpeak is -ve
        st.write("Input data with generated predictions:")
        OneHotEncoding = OneHotEncodeDf(df, data_loaded)
        df = OneHotEncoding.do_onehot_encoding() ## one hot encodes the categorical cols
        
        try: ## a crude way to deal with some unforseen errors with incorrect datatypes
            preds = get_preds(df, data_loaded) ## gets preds
        except ValueError as e:
            st.error('Warning some of the columns contain numerical values and not categorical ones, this needs to be addressed')
            return ## this just makes the script stop here after the error
        
        st.dataframe(preds)
        
        csv = preds.to_csv(index=False) ## turns data to csv 
        
        st.download_button( ## downloads the cleaned dataframe with preds
        label="Download data with predictions",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
            )
        
        if dropped_idx:
            st.write(f'The rows which were removed due to incorrect or missing values were: {dropped_idx}') 
            
        
        
if __name__ == '__main__':
    main()
