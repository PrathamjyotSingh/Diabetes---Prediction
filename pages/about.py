import streamlit as st
import pydeck as pdk

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=30,
        longitude=76,
        zoom=7,
        pitch=50,
    ))
)