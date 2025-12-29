from kiteconnect import KiteConnect

api_key = "bkgv59vaazn56c42"
api_secret = "sb1sxe6s2p9qbmajwnlfe8bxmxfzbzbf"
request_token = "9IlK75RVABdWC1BmOmSn3Nyohwnvrct5"

kite = KiteConnect(api_key=api_key)

data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

print("ACCESS TOKEN:", access_token)
