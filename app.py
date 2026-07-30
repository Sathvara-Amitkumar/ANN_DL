import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="🌸 Iris Flower Classification",
    layout="centered"
)

# Sidebar
# ==================================================

with st.sidebar:

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_dataset_scatterplot.svg",
        use_container_width=True
    )

    st.title("🤖 Model Information")

    st.markdown("---")

    st.metric("🎯 Test Accuracy", "96.00 %")
    st.metric("📊 Dataset", "Iris")
    st.markdown("---")
    st.subheader("🧠 Model Details")

    st.markdown("""
**Model :** Artificial Neural Network (ANN)

**Framework :** TensorFlow / Keras

**Optimizer :** Adam

**Loss Function :** 
- Categorical Crossentropy

**Activation Functions :**
- ReLU
- Softmax

**Preprocessing :**
- StandardScaler
- LabelEncoder
""")


    st.markdown(
    """
        <div style="text-align:center;">
            <a href="https://www.linkedin.com/in/sathvara-amitkumar-015783285/" target="_blank">
                <img src="https://img.icons8.com/?size=100&id=Ceo5wAFAEmmf&format=png&color=000000"
                    width="32">
            </a>
            &nbsp; &nbsp;
            <a href="https://github.com/Sathvara-Amitkumar/ANN_DL" target="_blank">
                <img src="https://img.icons8.com/?size=100&id=zehXPLJSAbBB&format=png&color=000000"
                    width="32">
            </a>
        </div>
    """,
    unsafe_allow_html=True,)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.caption("🚀 Developed by **Amitkumar Sathvara**")

# Load Model & Preprocessing Objects
# ==================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("iris_model.h5")


@st.cache_resource
def load_objects():
    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = joblib.load(f)

    return scaler, encoder

try:
    model = load_model()
    scaler, encoder = load_objects()
except Exception as e:
    st.error(f"❌ Error loading model or preprocessing files.\n\n{e}")
    st.stop()

# UI
st.title("🌸 Iris Flower Classification")

st.divider()


# Quick Example Buttons
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌸 Setosa Example", use_container_width=True):
        st.session_state.sepal_length = 5.1
        st.session_state.sepal_width = 3.5
        st.session_state.petal_length = 1.4
        st.session_state.petal_width = 0.2

with col2:
    if st.button("🌼 Versicolor Example", use_container_width=True):
        st.session_state.sepal_length = 5.9
        st.session_state.sepal_width = 3.0
        st.session_state.petal_length = 4.2
        st.session_state.petal_width = 1.5

with col3:
    if st.button("🌺 Virginica Example", use_container_width=True):
        st.session_state.sepal_length = 7.2
        st.session_state.sepal_width = 3.6
        st.session_state.petal_length = 6.1
        st.session_state.petal_width = 2.5



# Input Section
# ==================================================

col1, col2 = st.columns(2)

if "sepal_length" not in st.session_state:
    st.session_state.sepal_length = 5.1
    st.session_state.sepal_width = 3.5
    st.session_state.petal_length = 1.4
    st.session_state.petal_width = 0.2

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.sepal_length,
        step=0.1,
        format="%.1f"
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.sepal_width,
        step=0.1,
        format="%.1f"
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.petal_length,
        step=0.1,
        format="%.1f"
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=st.session_state.petal_width,
        step=0.1,
        format="%.1f"
    )

st.divider()


# Prediction
if st.button("🔍 Predict", use_container_width=True):

    try:
        # Prepare input
        input_data = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict probabilities
        prediction = model.predict(input_scaled, verbose=0)

        predicted_index = np.argmax(prediction)

        # Decode class label
        predicted_class = encoder.inverse_transform([predicted_index])[0]

        confidence = prediction[0][predicted_index] * 100

        st.success(f"### 🌼 Predicted Species : **{predicted_class}**")
        st.info(f"**Confidence:** {confidence:.2f} %")

        with st.expander("Prediction Probabilities"):
            for label, prob in zip(encoder.classes_, prediction[0]):
                st.write(f"**{label}:** {prob:.4f}")

    except Exception as e:
        st.error(f"Prediction failed.\n\n{e}")
