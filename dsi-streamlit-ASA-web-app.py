# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:28:54 2025

@author: steve
"""

# import libraries

import streamlit as st
import pandas as pd
import joblib

# load our model pipeline objects

abr_model = joblib.load("abr_superV_model.joblib")
asa_model = joblib.load("asa_superV_model.joblib")

# add title and instructions

st.title("ASA Prediction Model")
st.subheader("Enter the values for the following categories and submit for ASA prediction")

# Expected volume input form

pred_volume = st.number_input(
    label = "01. Enter the expected call volume",
    min_value = 100,
    max_value = 50000,
    value = 9261)

# AHT input form

AHT = st.number_input(
    label = "02. Enter the expected AHT in seconds",
    min_value = 100,
    max_value = 1000,
    value = 589)

# recommended scheduled phone hours input form

rec_sched_phone_hrs = st.number_input(
    label = "03. Enter the phone hrs recommended by Aspect (RevReq)",
    min_value = 10,
    max_value = 5000,
    value = 1624)

# actual scheduled phone hours input form

act_sched_phone_hrs = st.number_input(
    label = "04. Enter the actual phone hrs scheduled (SchwAdj)",
    min_value = 10,
    max_value = 5000,
    value = 1439)

# create variables for the new_data file

super_v = (pred_volume/act_sched_phone_hrs) * AHT/1000
net_staff = act_sched_phone_hrs - rec_sched_phone_hrs
if net_staff > -10:
    net_staff = -10

# submit inputs to model

if st.button("Submit For Prediction"):
    
    # store our data in a dataframe for ABR prediction
    new_abr_data = pd.DataFrame({"SuperV" : [super_v], "NetStaff" : [net_staff]})
    
    # apply ABR model pipeline to predict ABR
    abr = abr_model.predict(new_abr_data)
    
    # store our data in a dataframe for ABR prediction
    new_asa_data = pd.DataFrame({"ABR" : [abr], "SuperV" : [super_v], "NetStaff" : [net_staff]})
    
    # apply ABR model pipeline to predict ABR
    asa = asa_model.predict(new_asa_data)
    
    # apply model pipeline to the input data and extract probability prediction
    # pred_proba = asa_model.predict_proba(new_data)[0][1]
    # {pred_proba:.0%}
    
    # output prediction

    st.subheader(f"Based on these values, the model predicts a {asa-1} to {asa+1} minute ASA and an ABR of approximately {abr}%")

