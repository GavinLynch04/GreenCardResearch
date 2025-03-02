# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

pages = {
    "Home": [st.Page("pages/home.py", title="Home Page", default=True),],
    "Waiting Prediction": [
        st.Page("pages/multipage.py", title="Enter Your Info"),
        st.Page("pages/immigrationpage1dot5.py", title="View Wait time Breakdown"),
    ],
    "About This App": [
        st.Page("pages/immigrationpage2.py", title="About the Data"),
        st.Page("pages/immigrationpage3.py", title="About Us"),
        st.Page("pages/immigrationpage4.py", title="FAQs"),
        st.Page("pages/immigrationpage5.py", title="Additional Resources"),
    ],
}

st.navigation(pages).run()