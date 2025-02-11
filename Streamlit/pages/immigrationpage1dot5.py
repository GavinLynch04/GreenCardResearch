# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings
warnings.filterwarnings('ignore')

st.title('Feature Analysis')
st.divider()


st.subheader('Feature Importance')
tab_one,tab_two,tab_three,tab_four = st.tabs(['Decision Tree','Random Forest','AdaBoost','XGBoost'])
with tab_one:
      tab1, tab2 = st.tabs(["Individual Feature Importance","Aggregate Feature Importance"])
      with tab1:
        st.image('dt_importance_ind.png')
      with tab2:
         st.image('dt_importance.png')
with tab_two:
      tab1, tab2 = st.tabs(["Individual Feature Importance","Aggregate Feature Importance"])
      with tab1:
        st.image('rf_importance_ind.png')
      with tab2:
         st.image('rf_importance.png')
with tab_three:
      tab1, tab2 = st.tabs(["Individual Feature Importance","AggregateFeature Importance"])
      with tab1:
        st.image('ada_importance_ind.png')
      with tab2:
         st.image('ada_importance.png')
with tab_four:
      tab1, tab2 = st.tabs(["Individual Feature Importance","AggregateFeature Importance"])
      with tab1:
        st.image('xgb_importance_ind.png')
      with tab2:
         st.image('xgb_importance.png')


