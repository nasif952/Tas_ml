import streamlit as st


def has_ee_secret() -> bool:
    return "earthengine" in st.secrets


@st.cache_resource(show_spinner=False)
def init_earth_engine():
    """Initialize Earth Engine from Streamlit secrets."""
    import ee
    from google.oauth2 import service_account

    if not has_ee_secret():
        raise RuntimeError("No [earthengine] section found in Streamlit Secrets.")

    ee_secrets = st.secrets["earthengine"]

    required_keys = [
        "project", "type", "private_key_id", "private_key", "client_email",
        "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    ]
    missing = [key for key in required_keys if key not in ee_secrets]
    if missing:
        raise RuntimeError(f"Missing Earth Engine secret keys: {', '.join(missing)}")

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
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )

    ee.Initialize(credentials, project=ee_secrets["project"])
    return {
        "project": ee_secrets["project"],
        "client_email": ee_secrets["client_email"],
    }


def show_ee_connection_block():
    """Reusable Streamlit status block for EE pages."""
    try:
        ctx = init_earth_engine()
        st.success("Earth Engine connected.")
        c1, c2 = st.columns(2)
        c1.caption(f"Project: `{ctx['project']}`")
        c2.caption(f"Service account: `{ctx['client_email']}`")
        return True
    except Exception as exc:
        st.error("Earth Engine is not connected.")
        st.exception(exc)
        st.stop()
