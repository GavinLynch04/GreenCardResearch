# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.title('About')
st.divider()
st.header(':blue[Data]')
st.subheader('Sources')
st.markdown('The data for this project was obtained from two different sources. The primary dataset was obtained from publicly available records on the PERM program from the U.S. Department of Labor. The year of the decision date which the data was obtained for ranges from 2016-2024. All of the data comes directly from the application for an employment-based green card, with data inputs concerning both the applicant and the employer that is sponsoring them. The second dataset was provided by the United States Department of State - Bureau of Consular Affairs. This Visa Bulletin details the priority date of applications in which the U.S. is processing in any given month. Data was pulled for years ranging 2016 to 2024.')

st.subheader('Description')
st.markdown('The primary dataset contained 878,193 rows of data with 154 columns. The secondary dataset contained 108 rows and 12 columns. These datasets were combined using Python to create a column called Waiting Time which took into account the time betweeen the initial month/year the application was submitted and the month/year the U.S. began processing the application as well as the time it took to process and obtain a final decision. Through significant data cleaning and restructuring, the final combined dataset was cut down to 599,781 rows and 18 columns.')


st.divider()
st.header(':blue[About the Creators]')
st.subheader('Gavin Lynch')
st.markdown("Gavin Lynch is a student at California Polytechnic State University in San Luis Obispo, CA pursuing a Master of Science and Bachelor of Science in Computer Science. He has been working to improve upon Meghan's work on this project.")
st.subheader('Meghan Aerick')
st.markdown("Meghan Aerick is a student at California Polytechnic State University in San Luis Obispo, CA pursuing a Master of Science in Engineering Management and a Bachelor of Science in Aerospace Engineering. She created this application for her culminating graduate project.")
