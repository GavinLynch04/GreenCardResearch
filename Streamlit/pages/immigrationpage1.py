# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import warnings
warnings.filterwarnings('ignore')

st.title('ML Model Performance')
st.divider()

st.subheader('Model Peformance Comparison')
results = pd.read_csv('model_f1_scores.csv', index_col=0)
max_f1_index = results['f-1 Score Macro Average'].idxmax()
min_f1_index = results['f-1 Score Macro Average'].idxmin()

# Highlight the maximum and minimum values
def highlight_max_min_values(s):
    is_max = s == s.max()
    is_min = s == s.min()
    return ['background-color: limegreen' if max_v else ('background-color: orange' if min_v else '') for max_v, min_v in zip(is_max, is_min)]
styled_results = results.style.apply(highlight_max_min_values, subset=['f-1 Score Macro Average', 'f-1 Score Weighted Average', 'Accuracy'])
st.write('These ML models exhibited the following predictive performance on the test dataset.')
st.dataframe(styled_results)

st.divider()
st.subheader('Model Performance Metrics')
tab_one,tab_two,tab_three,tab_four = st.tabs(['Decision Tree','Random Forest','AdaBoost','XGBoost'])
with tab_one:
      tab3, tab4, tab5, tab6 = st.tabs(["Confusion Matrix (Train)","Confusion Matrix (Test)", "Classification Report (Train)", "Classification Report (Test)"])
      with tab3:
        st.image('dt_train_cm.png')
      with tab4:
        st.image('dt_test_cm.png')
      with tab5:
        df_train_class_report = pd.read_csv("df_train_class_report.csv", index_col=0)
        st.dataframe(df_train_class_report)
      with tab6:
        df_test_class_report = pd.read_csv("df_test_class_report.csv", index_col=0)
        st.dataframe(df_test_class_report)
with tab_two:
      tab3, tab4, tab5, tab6 = st.tabs(["Confusion Matrix (Train)","Confusion Matrix (Test)", "Classification Report (Train)", "Classification Report (Test)"])
      with tab3:
        st.image('rf_train_cm.png')
      with tab4:
        st.image('rf_test_cm.png')
      with tab5:
        rf_train_class_report = pd.read_csv("rf_train_class_report.csv", index_col=0)
        st.dataframe(rf_train_class_report)
      with tab6:
        rf_test_class_report = pd.read_csv("rf_test_class_report.csv", index_col=0)
        st.dataframe(rf_test_class_report)
with tab_three:
      tab3, tab4, tab5, tab6 = st.tabs(["Confusion Matrix (Train)","Confusion Matrix (Test)", "Classification Report (Train)", "Classification Report (Test)"])
      with tab3:
        st.image('ada_train_cm.png')
      with tab4:
        st.image('ada_test_cm.png')
      with tab5:
        ada_train_class_report = pd.read_csv("ada_train_class_report.csv", index_col=0)
        st.dataframe(ada_train_class_report)
      with tab6:
        ada_test_class_report = pd.read_csv("ada_test_class_report.csv", index_col=0)
        st.dataframe(ada_test_class_report)
with tab_four:
      tab3, tab4, tab5, tab6 = st.tabs(["Confusion Matrix (Train)","Confusion Matrix (Test)", "Classification Report (Train)", "Classification Report (Test)"])
      with tab3:
        st.image('xgb_train_cm.png')
      with tab4:
        st.image('xgb_test_cm.png')
      with tab5:
        xgb_train_class_report = pd.read_csv("xgb_train_class_report.csv", index_col=0)
        st.dataframe(xgb_train_class_report)
      with tab6:
        xgb_test_class_report = pd.read_csv("xgb_test_class_report.csv", index_col=0)
        st.dataframe(xgb_test_class_report)


