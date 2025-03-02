import streamlit as st
st.title('United States Employment-Based Green Card Waiting Time Prediction: A Machine Learning App')
st.divider()

st.image('./Streamlit/greencard.jpg', width = 700)

st.markdown(
    'As there is much uncertainty for U.S. green card applicants on how long they must wait before obtaining a green card, this application was created to provide a predicted waiting time range. Machine learning models were trained from past application data from green cards that were approved and processed. While this application is not perfect, it does provide an approximation of waiting time to allow applicants or potential applicants to better plan for their future.')

st.markdown(
    "This application was created as a graduate research project at Cal Poly in San Luis Obispo, California. More about the creators can be found on the 'About' page.")

st.markdown("For any questions, please refer to the 'FAQs' page.")

st.markdown('Thank you for using this application!')