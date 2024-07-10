import numpy as np
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import streamlit as st
from tagger_tools import compute_tof_from_data
from bokeh.charts import Histogram


class TaggerVizualizer:
    def __init__(self, read_data: pd.DataFrame):
        self.read_data = read_data
        
    def tof_histogram(self, data):
        tof_df = compute_tof_from_data(data)
        tofs = tof_df["tof"].values * 1e6
        mu = np.mean(tofs)
        sigma = np.std(tofs)
        hist = Histogram(tofs, bins=50, mu=mu, sigma=sigma,
                 title="ToF", ylabel="Frequency", xlabel="Time [µs]", legend="top_left",
                 width=400, height=350, notebook=True)
        st.bokeh_chart(hist)