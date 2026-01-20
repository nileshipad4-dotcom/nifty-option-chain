import streamlit as st
from kiteconnect import KiteConnect

st.title("Zerodha Kite Access Token Generator")

api_key = "48u6v9ed04hmb7ya"
api_secret = "gouxec13l4jrl3mhp65cwuxrsyvj48cl"

st.markdown(
    '<a href="https://kite.zerodha.com/connect/login?api_key=48u6v9ed04hmb7ya&v=3" target="_blank">Login to Zerodha</a>',
    unsafe_allow_html=True
)

raw_input = st.text_input("Enter Request Token or Redirect URL", type="password")

def extract_request_token(text):
    if "request_token=" in text:
        return text.split("request_token=", 1)[1].split("&", 1)[0].strip()
    elif "=" in text:
        return text.rsplit("=", 1)[-1].strip()
    else:
        return text.strip()

request_token = extract_request_token(raw_input) if raw_input else ""

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
        st.warning("Please enter a request token or URL")
