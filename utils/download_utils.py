import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def dataframe_downloads(df: pd.DataFrame, base_filename: str, label_prefix: str = "Download") -> None:
    """Render local browser download buttons for a dataframe."""
    if df is None or df.empty:
        st.info("No table is available to download yet.")
        return

    c1, c2, c3 = st.columns(3)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    c1.download_button(
        f"{label_prefix} CSV",
        csv_bytes,
        file_name=f"{base_filename}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
    c2.download_button(
        f"{label_prefix} JSON",
        json_bytes,
        file_name=f"{base_filename}.json",
        mime="application/json",
        use_container_width=True,
    )

    txt_bytes = df.to_string(index=False).encode("utf-8")
    c3.download_button(
        f"{label_prefix} TXT",
        txt_bytes,
        file_name=f"{base_filename}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def dict_downloads(data: Dict[str, Any], base_filename: str, label_prefix: str = "Download metadata") -> None:
    """Render JSON/TXT download buttons for small metadata dictionaries."""
    if not data:
        st.info("No metadata is available to download yet.")
        return

    c1, c2 = st.columns(2)
    json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
    txt_bytes = "\n".join([f"{k}: {v}" for k, v in data.items()]).encode("utf-8")

    c1.download_button(
        f"{label_prefix} JSON",
        json_bytes,
        file_name=f"{base_filename}.json",
        mime="application/json",
        use_container_width=True,
    )
    c2.download_button(
        f"{label_prefix} TXT",
        txt_bytes,
        file_name=f"{base_filename}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def text_download(text: str, filename: str, label: str, mime: str = "text/plain") -> None:
    st.download_button(
        label,
        str(text).encode("utf-8"),
        file_name=filename,
        mime=mime,
        use_container_width=True,
    )


def repository_text_file_download(path: str, label: Optional[str] = None) -> None:
    """Download a text/markdown file that exists inside the deployed repo."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        st.warning(f"File not found in deployed app: {path}")
        return

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    filename = file_path.name
    mime = "text/markdown" if filename.lower().endswith(".md") else "text/plain"
    text_download(text, filename, label or f"Download {filename}", mime=mime)


def list_repository_files(folder: str, suffixes=(".md", ".txt", ".csv", ".json")):
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(str(p).replace("\\", "/"))
    return sorted(files)
