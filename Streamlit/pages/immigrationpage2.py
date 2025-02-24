# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.title('Descriptive Analytics') 
st.divider()

st.header('Analysis of Data')
st.markdown('The dataset was analyzed using graphs and plots to visualize the data. The following notable characteristics of the dataset were observed:')
st.markdown("- The majority of wait times were grouped in either the **0-2.5 years** category or the **>10 years** category.")
st.markdown("- More than half of the dataset is for applicants whose country of citizenship is **India**.")
st.markdown("- The top 5 countries with the most data are **India, China, Canada, South Korea,** and **Mexico**.")


st.image('Streamlit/PIEE2.png')
st.image('Streamlit/PIEE.png')

st.header('Analysis of Features Impacting Waiting Times')
st.markdown('An analysis of the dataset was performed to identify major factors influencing visa wait times. Observed trends are as follows:')
st.markdown("- The top 5 countries with the **highest average waiting times** are **India, China, Marshall Islands, Philippines,** and **Namibia**.")
st.markdown("- A **higher prevailing wage level** (Level IV being the highest), indicative of a higher-level position with more complex duties, was associated with **higher wait times**.")
st.markdown("- Foreign workers with **more education** up to a Master's Degree had **higher wait times** than those with lower education. On the other hand, a **Doctorate degree decreased the wait time** in comparison to a Bachelor's or Master's.")
st.markdown("- Applicants with employers who **required job experience** had **higher average wait times** than those that did not.")
st.markdown("- ")

st.image('Streamlit/average_waiting_time_map.svg')
st.image('Streamlit/PWlevel.svg')
st.image('Streamlit/eduworker.svg')
st.image('Streamlit/experience.svg')