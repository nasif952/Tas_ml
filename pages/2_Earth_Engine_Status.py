import streamlit as st

st.set_page_config(
    page_title="Earth Engine Status · Tasmania Flood Intelligence",
    page_icon="✅",
    layout="wide",
)

st.title("✅ Earth Engine Connection Status")
st.caption("Use this page after adding Streamlit Secrets to confirm that the deployed app can authenticate with Google Earth Engine.")

try:
    import ee
    from google.oauth2 import service_account
except Exception as import_error:
    st.error("Required Earth Engine packages are not installed yet.")
    st.exception(import_error)
    st.stop()


@st.cache_resource(show_spinner=False)
def init_earth_engine():
    ee_secrets = st.secrets["earthengine"]

    service_account_info = {
        "type": ee_secrets["type"],
        "project_id": ee_secrets["project"],
        "private_key_id": ee_secrets["private_key_id"],
        "private_key": ee_secrets["private_key"],
        "client_email": ee_secrets["client_email"],
        "client_id": ee_secrets["client_id"],
        "auth_uri": ee_secrets["auth_uri"],
        "token_uri": ee_secrets["token_uri"],
        "auth_provider_x509_cert_url": ee_secrets["auth_provider_x509_cert_url"],
        "client_x509_cert_url": ee_secrets["client_x509_cert_url"],
    }

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )

    ee.Initialize(credentials, project=ee_secrets["project"])
    return True


if "earthengine" not in st.secrets:
    st.error("No [earthengine] section found in Streamlit Secrets.")
    st.stop()

st.subheader("Secret summary")
st.write("Project:", st.secrets["earthengine"].get("project", "missing"))
st.write("Client email:", st.secrets["earthengine"].get("client_email", "missing"))

if st.button("Test Earth Engine Connection", type="primary"):
    with st.spinner("Connecting to Earth Engine..."):
        try:
            init_earth_engine()
            img = ee.Image("USGS/SRTMGL1_003")
            test_value = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([147.33, -42.88]).buffer(1000),
                scale=90,
                maxPixels=1e6,
            ).getInfo()
            st.success("Earth Engine connected successfully.")
            st.json(test_value)
        except Exception as e:
            st.error("Earth Engine connection failed.")
            st.exception(e)
            st.info("Check that the service account key is valid, Earth Engine API is enabled, the Cloud project is registered for Earth Engine, and the service account has Earth Engine permissions.")
