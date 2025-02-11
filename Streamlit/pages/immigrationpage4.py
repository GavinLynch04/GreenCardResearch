# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df_naics = pd.read_excel('naics.xlsx')

st.title('Frequently Asked Questions')
st.divider()
st.subheader('How do I input my data?')
st.markdown('You can upload either a csv file in the specified format or use the drop-down entry options to enter your data and obtain the result.')
st.divider()
st.subheader('What is a NAICS Code?')
st.markdown('A NAICS Code is a classification within the North American Industry Classification System. The NAICS System was developed for use by Federal Statistical Agencies for the collection, analysis and publication of statistical data related to the US Economy. The codes are obtained from the U.S. Census Bureau. Below is a table describing each NAICS code.')
df_naics.set_index('NAICS Code', inplace=True)
st.table(df_naics)
st.divider()
st.subheader('How accurate is the prediction?')
st.markdown("The models are tuned for optimal performance. The classification report on the 'ML Model Performance' page shows the obtained f1, recall, and precision scores for each model. The confusion matrices for the training and testing data sets are also displayed.")
st.divider()
st.subheader('What countries of citizenship does this work for?')
st.markdown('The model is built based off data from all 195 countries in the world, so it is valid for any country of citizenship.')
st.divider()
st.subheader('What do all the plots mean?')
st.markdown('The feature importance plot shows the extent to which each input feature impacts the waiting time prediction. The aggregate feature importance plot displays the combined importances for each category. The confusion matrices show the predicted versus actual values for waiting times for both the testing and training data. The classification reports display the resulting f1, recall, and precision scores for each category of waiting times for both training and testing data. This explains the performance of the selected model.')
st.divider()


