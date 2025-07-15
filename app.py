import streamlit as st
import tempfile
import json
import cv2
import os

from ultralytics import YOLO
from config import DETECTION_CONF, OUTPUT_JSON

from utils.detection import detect_players
from utils.matching import match_players
from utils.visualization import draw_boxes

from config import *


st.set_page_config(page_title="Player Matcher", layout="centered")

st.markdown("<h1 style='text-align: center;'>Player Matcher</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Match players between broadcast and tacticam</p>", unsafe_allow_html=True)

st.subheader("Upload Files")

col1, col2, col3 = st.columns(3)
with col1:
    uploaded_model = st.file_uploader("YOLOv8 Model (.pt)", type=["pt"], label_visibility="collapsed")
    st.caption("Upload your trained YOLOv8 model")

with col2:
    uploaded_broadcast = st.file_uploader("Broadcast Video (.mp4)", type=["mp4"], label_visibility="collapsed")
    st.caption("Upload broadcast camera footage")

with col3:
    uploaded_tacticam = st.file_uploader("Tacticam Video (.mp4)", type=["mp4"], label_visibility="collapsed")
    st.caption("Upload tacticam camera footage")

st.divider()

run_button = st.button("Run Player Matching", use_container_width=True)

if run_button and uploaded_model and uploaded_broadcast and uploaded_tacticam:
    with st.spinner("Running detection and player matching..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
            tmp_model.write(uploaded_model.read())
            model_path = tmp_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_broadcast:
            tmp_broadcast.write(uploaded_broadcast.read())
            broadcast_path = tmp_broadcast.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_tacticam:
            tmp_tacticam.write(uploaded_tacticam.read())
            tacticam_path = tmp_tacticam.name

        try:
            model = YOLO(model_path)
            model.to(DEVICE)
            broadcast_dets, broadcast_frames = detect_players(broadcast_path, model, DETECTION_CONF)
            tacticam_dets, tacticam_frames = detect_players(tacticam_path, model, DETECTION_CONF)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

        mapping = match_players(tacticam_dets, broadcast_dets)

        st.session_state.broadcast_dets = broadcast_dets
        st.session_state.tacticam_dets = tacticam_dets
        st.session_state.broadcast_frames = broadcast_frames
        st.session_state.tacticam_frames = tacticam_frames
        st.session_state.mapping = mapping
        st.session_state.mapping_json = json.dumps(mapping, indent=2)

        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(mapping, f, indent=2)

        st.success("Player Matching Complete!")


if "broadcast_dets" in st.session_state:
    st.subheader("Player Mapping Output")
    st.code(st.session_state.mapping_json, language="json")

    st.download_button("Download Mapping as JSON",
                       st.session_state.mapping_json,
                       file_name="player_mapping.json",
                       mime="application/json")

    st.divider()
    st.subheader("Frame Viewer")

    max_frame = max(st.session_state.broadcast_frames.keys())
    selected_frame = st.slider("Select Frame Index", 0, max_frame, 0, step=1)

    b_frame = st.session_state.broadcast_frames.get(selected_frame)
    t_frame = st.session_state.tacticam_frames.get(selected_frame)

    if b_frame is not None and t_frame is not None:
        col1, col2 = st.columns(2)

        with col1:
            b_annotated = draw_boxes(b_frame.copy(), st.session_state.broadcast_dets[selected_frame], "broadcast")
            st.image(cv2.cvtColor(b_annotated, cv2.COLOR_BGR2RGB),
                     caption=f"Broadcast Frame {selected_frame}")

        with col2:
            t_annotated = draw_boxes(t_frame.copy(), st.session_state.tacticam_dets[selected_frame], "tacticam")
            st.image(cv2.cvtColor(t_annotated, cv2.COLOR_BGR2RGB),
                     caption=f"Tacticam Frame {selected_frame}")
    else:
        st.warning("Frame not found for the selected index.")
else:
    st.info("Upload model and both videos, then click 'Run Player Matching' to begin.")
