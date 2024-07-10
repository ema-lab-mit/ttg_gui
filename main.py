import streamlit as st
from tagger import Tagger
from pags.Setup_Parameters import Setup_Parameters
from pags.Display_Events_ToF import Display_Events_ToF

st.set_page_config(
    page_title="EMA Beamline Tagger GUI",
    page_icon="⌂",
)



def initialize_tagger():
    if 'tagger_card' not in st.session_state:
        st.session_state.tagger_card = Tagger()
        st.session_state.tagger_card.set_trigger_falling()
        st.session_state.tagger_card.set_trigger_level(-0.5)
        st.session_state.tagger_card.start_reading()
        st.session_state.initialized = True

def main():
    st.write("# EMA Beamline Tagger GUI")
    st.markdown(
        """
        Streamlit is an open-source app framework.
        """
    )

    # Initialize the tagger card if not already initialized
    initialize_tagger()

    # Sidebar navigation
    page = st.sidebar.radio("Select Page", ["Setup Parameters", "Display Events and ToF"])

    if page == "Setup Parameters":
        setup_parameters = Setup_Parameters()
        setup_parameters.render()
    elif page == "Display Events and ToF":
        display_events = Display_Events_ToF()
        display_events.render()

if __name__ == "__main__":
    main()
