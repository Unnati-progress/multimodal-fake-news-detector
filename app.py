import streamlit as st
import tempfile
import os

from Multimodel import final_verify_multimodal

st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

st.title("Fake News Detection System")
st.write("Enter a news claim or upload an image/video to check whether the news is real or fake.")

claim_text = st.text_area(
    "Enter news claim / text",
    placeholder="Example: Today India will observe black day because Modi ji died"
)

uploaded_image = st.file_uploader("Upload news image", type=["jpg", "jpeg", "png"])
uploaded_video = st.file_uploader("Upload news video", type=["mp4", "avi", "mov", "mkv"])

image_path = None
video_path = None

if uploaded_image is not None:
    suffix = os.path.splitext(uploaded_image.name)[1] or ".jpg"
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_image.write(uploaded_image.read())
    temp_image.close()
    image_path = temp_image.name

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

if uploaded_video is not None:
    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_video.write(uploaded_video.read())
    temp_video.close()
    video_path = temp_video.name

    st.video(video_path)

if st.button("Verify News"):
    if not claim_text.strip() and image_path is None and video_path is None:
        st.warning("Please enter text or upload image/video.")
    else:
        with st.spinner("Analyzing news... Please wait."):
            result = final_verify_multimodal(
                claim_text=claim_text,
                image_path=image_path,
                video_path=video_path
            )

        final_decision = result.get("final_decision", result.get("final", "Unknown"))
        reason = result.get("reason", "No explanation available.")

        decision_lower = str(final_decision).lower()

        st.subheader("Final Result")

        if decision_lower == "real":
            st.success("REAL NEWS")

        elif decision_lower == "fake":
            st.error("FAKE NEWS")

        elif decision_lower == "suspicious":
            st.warning("SUSPICIOUS NEWS")

        else:
            st.info("UNKNOWN RESULT")
        st.subheader("Confidence")

        text_conf = result.get("text_conf", 0)
        image_conf = result.get("image_conf", 0)
        video_conf = result.get("video_conf", 0)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Text Confidence",
                f"{round(text_conf * 100, 2)}%"
            )

        with col2:
            if image_path is not None:
                st.metric(
                    "Image Confidence",
                    f"{round(image_conf * 100, 2)}%"
                )
            else:
                st.metric("Image Confidence", "Not uploaded")

        with col3:
            if video_path is not None:
                st.metric(
                    "Video Confidence",
                    f"{round(video_conf * 100, 2)}%"
                )
            else:
                st.metric("Video Confidence", "Not uploaded")

        st.subheader("Explanation")
        st.write(reason)

        st.subheader("Evidence Sources")

        evidence = result.get("top_evidence", [])

        if evidence:
            for i, e in enumerate(evidence, start=1):
                title = e.get("title", "Evidence Source")
                description = e.get("description", "No description available.")
                url = e.get("url", "")
                score = e.get("score", 0)

                st.markdown(f"### {i}. {title}")
                st.write(description)

                if url:
                    st.markdown(f"[Open Source]({url})")

                st.write(f"Evidence Match: {round(score * 100, 2)}%")
                st.divider()
        else:
            st.info("No evidence source found.")

        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        if video_path and os.path.exists(video_path):
            os.remove(video_path)