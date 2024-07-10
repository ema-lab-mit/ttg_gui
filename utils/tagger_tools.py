import streamlit as st
import numpy as np
import pandas as pd
import time

def convert_to_stoptime(t):
    # 30000 -> ~15 us
    convertion = 30_000/(15e-6) #p / s
    return convertion * t

def time_to_flops(t):
    quatization = 100e-12 # seconds / flops
    return t / quatization

def flops_to_time(ft):
    quatization = 100e-12 # seconds / flops
    return ft * quatization
    
    
def compute_tof_from_data(data: pd.DataFrame):
    latest_trigger_time = 0
    output_data = {"tof": [], "timestamp": []}
    for index, d in data.iterrows():
        is_trigger = d.channels == -1
        if is_trigger:
            latest_trigger_time = d.timestamp
        else:
            tof = d["timestamp"] - latest_trigger_time
            output_data["tof"].append(tof)
            output_data["timestamp"].append(d["timestamp"])
    return pd.DataFrame(output_data)