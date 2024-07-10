import os
import sys
import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px

# Set paths and import necessary modules
this_path = os.path.abspath(__file__)
father_path = "C:\\Users\\EMALAB\\Desktop\\TW_DAQ"
sys.path.append(father_path)
from TimeTaggerDriver_isolde.timetagger4 import TimeTagger as tg
from beamline_tagg_gui.utils.tagger_tools import time_to_flops, flops_to_time

# Streamlit page configuration
st.set_page_config(
    page_title="EMA Beamline Tagger GUI",
    page_icon="⌂",
)

# Initialization parameters
initialization_params = {
    "trigger": {
        "channels": [True, True, True, True],
        "levels": [-0.5, -0.5, -0.5, -0.5],
        "types": [False, False, False, False], 
        "starts": [int(time_to_flops(1e-6)) for _ in range(4)],
        "stops": [int(time_to_flops(20e-6)) for _ in range(4)],
    }
}

N_CHANNELS = ["A", "B", "C", "D"]
SAVE_PATH = "C:\\Users\\EMALAB\\Music\\TDC.csv"

@st.cache_resource
def initialize_tagger() -> object:
    default_channels = initialization_params['trigger']['channels']
    default_levels = initialization_params['trigger']['levels']
    default_types = initialization_params['trigger']['types']
    kwargs = {
        'trigger_level': -0.5,
        'trigger_rising': False
    }
    for i, info in enumerate(zip(default_channels, default_levels, default_types,
                                initialization_params['trigger']['starts'],
                                initialization_params['trigger']['stops'])):
        ch, l, t, stt, sp = info
        kwargs[f'channel_{i}_used'] = ch
        kwargs[f'channel_{i}_level'] = l
        kwargs[f'channel_{i}_rising'] = t
        kwargs[f'channel_{i}_start'] = stt
        kwargs[f'channel_{i}_stop'] = sp

    kwargs['index'] = 0

    if 'tagger_card' not in st.session_state:
        st.session_state.tagger_card = tg(**kwargs)
        st.session_state.tagger_card.startReading()
        st.session_state.initialized = True
    
    return st.session_state.tagger_card

class TaggerInterface:
    def __init__(self, tagger_card: object = None):
        self.index = 0
        self.started = False
        self.card = tagger_card

    def set_trigger_level(self, level):
        self.trigger_level = level

    def set_trigger_rising(self):
        self.set_trigger_type(type='rising')

    def set_trigger_falling(self):
        self.set_trigger_type(type='falling')

    def set_trigger_type(self, type='falling'):
        self.trigger_type = type == 'rising'

    def enable_channel(self, channel):
        self.channels[channel] = True

    def disable_channel(self, channel):
        self.channels[channel] = False

    def set_channel_level(self, channel, level):
        self.levels[channel] = level

    def set_channel_rising(self, channel):
        self.set_type(channel, type='rising')

    def set_channel_falling(self, channel):
        self.set_type(channel, type='falling')

    def set_type(self, channel, type='falling'):
        self.type[channel] = type == 'rising'

    def set_channel_window(self, channel, start=0, stop=600000):
        self.starts[channel] = start
        self.stops[channel] = stop

    def get_data(self, timeout=2):
        start = time.time()
        while time.time() - start < timeout:
            status, data = self.card.getPackets()
            if status == 0:
                if not data:
                    print('no data')
                    return None
                else:
                    return data
            elif status == 1:
                time.sleep(0.000001)
            else:
                raise ValueError
        return None

    def get_status(self):
        status, _ = self.card.getPackets()
        return status

    def status_to_string(self, status):
        if status == 0:
            return "Data available"
        elif status == 1:
            return "No data available"
        else:
            return "Error"

    def stop(self):
        if self.card is not None:
            if self.started:
                self.card.stopReading()
                self.started = False
            self.card.stop()
            self.card = None
            
    def build_dataframe(self, data):
        dataset = {"bunch": [], "n_events": [], "channels": [], "timestamp": []}
        for d in data:
            dataset['bunch'].append(d[0])
            dataset['channels'].append(d[-2])
            dataset['n_events'].append(d[1])
            dataset['timestamp'].append(d[-1])
        df = pd.DataFrame(dataset)
        df.timestamp = df.timestamp.apply(flops_to_time)
        return df

tg = initialize_tagger()
tagger = TaggerInterface(tg)

st.title("EMA Beamline Tagger GUI")

class SetupParameters:
    def __init__(self, tagger_interface: TaggerInterface):
        self.tagger_int = tagger_interface

    def update_parameters(self, new_params: dict):
        self.tagger_int.set_trigger_level(new_params['level'])
        if new_params['rising']:
            self.tagger_int.set_trigger_rising()
        else:
            self.tagger_int.set_trigger_falling()

        for channel in [1, 2, 3, 4]:
            if channel in new_params['active_channels']:
                self.tagger_int.enable_channel(channel-1)
            else:
                self.tagger_int.disable_channel(channel-1)
            self.tagger_int.set_channel_level(channel-1, new_params['level'])
            self.tagger_int.set_channel_window(channel-1, 
                                               start=time_to_flops(new_params['start_time']), 
                                               stop=time_to_flops(new_params['stop_time']))

    def params_to_session_state(self):
        st.session_state['parameters'] = self.parameters

    def render(self):
        st.title("Tagger Parameter Setup")
        self.parameters = {
            "start_time": st.number_input("Start Time (s)", min_value=0.0, value=5e-6, format="%e"),
            "stop_time": st.number_input("Stop Time (s)", min_value=5e-6, value=50e-6, format="%e"),
            "active_channels": st.multiselect("Active Channels", [1, 2, 3, 4], default=[1, 2, 3, 4]),
            "level": st.slider("Level (V)", min_value=-5.0, max_value=5.0, value=-0.5),
            "rising": st.checkbox("Rising Edge Trigger", value=False),
        }
        
        if st.button("Update Parameters", key="update_params"):
            self.update_parameters(self.parameters)
            self.params_to_session_state()
            st.success("Parameters Updated")

param_setter = SetupParameters(tagger)

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

class TaggerVisualizer:
    def __init__(self):
        self.rolling_window = 1

    def tof_histogram(self, data):
        tof_df = compute_tof_from_data(data)
        tofs = tof_df["tof"].values * 1e6
        if not len(tofs):
            return
        mu = np.mean(tofs)
        hist = px.histogram(tof_df, x="tof", nbins=100, title="TOF distribution", 
                            labels={"tof": "Time of flight (s)", "timestamp": "Events"})
        hist.add_shape(type='line', x0=mu, x1=mu, y0=0, y1=0.5, line=dict(color='red', width=2))
        st.plotly_chart(hist)

    def events_per_bunch(self, data, rolling_window=1):
        foo_df = data.copy().drop_duplicates(subset=["bunch"])
        # Rolling window sums the number of events per bunch in a window of size rolling_window of bunches
        rolled_df = foo_df.groupby("bunch").n_events.sum().rolling(rolling_window).sum()
        bunches = rolled_df.index
        num_events = rolled_df.values
        px.line(x=bunches, y=num_events, title="Events per Bunch", labels={"x": "Bunch", "y": "Number of Events"})


viz = TaggerVisualizer()

class DisplayEventsToF:
    def __init__(self, tagger_interface: object):
        self.df = pd.DataFrame({"bunch": [], "n_events": [], "channels": [], "timestamp": []})
        self.tagger_interface = tagger_interface
        self.runner = True
        self.plot_placeholder = st.empty()
        
    def looper_step(self, sleeper=0.1):
        data = self.tagger_interface.get_data()
        if data:
            self.df = pd.concat([self.df, self.tagger_interface.build_dataframe(data)])
        time.sleep(sleeper)
        
    def stop(self):
        self.runner = False

    def render(self):
        self.plot_placeholder.empty()
        refreshing_rate = st.number_input("Refresh Rate (s)", min_value=0.01, max_value=1., format="%f", key="refresh_rate")
        st.write('## Display Events ToF')
        stop_button = st.button("Stop", key="stop_tof_display")
        self.runner = True
        with st.form(key="rolling_window_form"):
            rolling_window = st.slider("Rolling Window", min_value=1, max_value=100, value=1, key="rolling_window")
            submit_button = st.form_submit_button("Update Rolling Window")
        while self.runner:
            self.looper_step(refreshing_rate)
            with self.plot_placeholder.container():
                if self.df.empty >=1:
                    st.warning("No data available!")
                    status = self.tagger_interface.get_status()
                    st.write("Status: ", self.tagger_interface.status_to_string(status))
                else:
                    viz.tof_histogram(self.df)
                    viz.events_per_bunch(self.df, rolling_window=rolling_window)
            if stop_button:
                self.stop()
                break
            time.sleep(0.01)

tof_display = DisplayEventsToF(tagger)

def load_tabs():
    possible_pages = ["Tagger Setup", "Tagger", "Laser", "Scan"]
    tab1, tab2, tab3, tab4 = st.tabs(possible_pages)
    placeholder = st.empty()
    with tab1:
        with placeholder.container():
            param_setter.render()
    with tab2:
        with placeholder.container():
            tof_display.render()
    with tab3:
        st.write("TODO: Laser tab content... implement the same UI here, kinda naive")
    with tab4:
        st.write("TODO: Scan tab content")

load_tabs()
