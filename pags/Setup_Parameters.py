import streamlit as st

class Setup_Parameters:

    def setup_tagger(self, tagger_card, parameters):
        tagger_card.set_trigger_level(parameters["level"])

        if parameters["rising"]:
            tagger_card.set_trigger_rising()
        else:
            tagger_card.set_trigger_falling()

        for channel in range(4):
            if channel + 1 in parameters["active_channels"]:
                tagger_card.enable_channel(channel)
            else:
                tagger_card.disable_channel(channel)

        for channel in range(4):
            tagger_card.set_channel_window(channel, start=int(self.time_to_flops(parameters["start_time"])), stop=int(self.time_to_flops(parameters["stop_time"])))

        tagger_card.init_card()
        tagger_card.start_reading()

    def render(self):
        st.title("Tagger Parameter Setup")

        parameters = {
            "start_time": st.number_input("Start Time (s)", min_value=0.0, value=1e-6, format="%e"),
            "stop_time": st.number_input("Stop Time (s)", min_value=0.0, value=20e-6, format="%e"),
            "active_channels": st.multiselect("Active Channels", [1, 2, 3, 4], default=[1, 2, 3, 4]),
            "level": st.slider("Level (V)", min_value=-5.0, max_value=5.0, value=-0.5),
            "rising": st.checkbox("Rising Edge Trigger", value=False),
        }

        if st.button("Update Tagger Parameters"):
            self.setup_tagger(st.session_state.tagger_card, parameters)
            st.success("Tagger parameters updated successfully!")

        st.write("Current Parameters:", parameters)
