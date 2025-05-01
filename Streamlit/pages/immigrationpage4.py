# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df_naics = pd.read_excel('Streamlit/naics.xlsx')

st.title('Frequently Asked Questions')
st.divider()
st.subheader('How do I input my data?')
st.markdown('You can use the drop-down entry options to enter your data into the form, and then obtain the prediction and feature analysis.')
st.divider()
st.subheader('What is a NAICS Code?')
st.markdown('A NAICS Code is a classification within the North American Industry Classification System. The NAICS System was developed for use by Federal Statistical Agencies for the collection, analysis and publication of statistical data related to the US Economy. The codes are obtained from the U.S. Census Bureau. Below is a table describing each NAICS code.')
df_naics.set_index('NAICS Code', inplace=True)
st.table(df_naics)
st.divider()
st.subheader('How accurate is the prediction?')
st.markdown("The models are tuned for optimal performance. The average RMSE (Root Mean Squared Error) for the model used in this application was 0.73 years. This means the average prediction was 0.73 years (~8 months) away from the true value. Using the prediction in tandem with the confidence interval will give you a better idea of how accurate your prediction is.")
st.divider()
st.subheader('What countries of citizenship does this work for?')
st.markdown('The model is built based off data from all 195 countries in the world, so it is valid for any country of citizenship.')
st.divider()
st.subheader('What do all the plots mean?')
st.markdown('The feature importance plot shows the extent to which each input feature impacts the waiting time prediction. The aggregate feature importance plot displays the combined importances for each category.')
st.divider()


