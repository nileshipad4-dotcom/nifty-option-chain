# https://kite.zerodha.com/connect/login?api_key=bkgv59vaazn56c42&v=3

import streamlit as st
from kiteconnect import KiteConnect

st.title("Zerodha Kite Access Token Generator")

api_key = "bkgv59vaazn56c42"
api_secret = "sb1sxe6s2p9qbmajwnlfe8bxmxfzbzbf"

st.markdown(
    '<a href="https://kite.zerodha.com/connect/login?api_key=bkgv59vaazn56c42&v=3" target="_blank">Hi</a>',
    unsafe_allow_html=True
)

raw_input = st.text_input("Enter Request Token", type="password")

request_token = (
    raw_input.rsplit("=", 1)[-1].strip()
    if raw_input
    else ""
)


if st.button("Generate Access Token"):
    if request_token:
        try:
            kite = KiteConnect(api_key=api_key)
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]

            st.success("Access Token Generated Successfully")
            st.code(access_token)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a request token")
