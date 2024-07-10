import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit
import os
import sys
import streamlit as st
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from beamline_tagg_gui.utils.plotter import TaggerVizualizer

class Display_Events_ToF:
    def init(self, tagger_interface: object):
        self.data = pd.DataFrame()
        
    def create_display(self):
        st.write('## Display Events ToF')
        cols = [1,1]
        
    def looper()