import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Load color dataset
@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
    return df

# Get closest color name using Manhattan distance in RGB space
def get_closest_color_name(R, G, B, df):
    min_dist = float('inf')
    closest_color = ""
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            closest_color = df.loc[i, "color_name"]
    return closest_color

# Streamlit app starts here
def main():
    st.title("🎨 Color Detection from Image")
    st.write("Upload an image and click on it to detect the color name and RGB values.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        df = load_colors()

        st.write("Click anywhere on the image to detect color.")
        st.write("Note: Use the coordinates below to test manually if Streamlit doesn't support clicks.")

        # Manual coordinate input
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("X Coordinate", min_value=0, max_value=image_np.shape[1]-1, step=1)
        with col2:
            y = st.number_input("Y Coordinate", min_value=0, max_value=image_np.shape[0]-1, step=1)

        if st.button("Detect Color"):
            pixel = image_np[int(y), int(x)]
            R, G, B = int(pixel[0]), int(pixel[1]), int(pixel[2])
            color_name = get_closest_color_name(R, G, B, df)

            st.markdown(f"**Color Name:** {color_name}")
            st.markdown(f"**RGB:** ({R}, {G}, {B})")

            st.markdown("### Detected Color:")
            st.markdown(
                f"<div style='width:100px;height:100px;background-color:rgb({R},{G},{B});border-radius:10px'></div>",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()
