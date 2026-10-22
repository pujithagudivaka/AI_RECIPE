
from pathlib import Path
import base64

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page settings
st.set_page_config(
    page_title="AI Recipe Chef",
    page_icon="🍲",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).parent
BG_PATH = BASE_DIR / "assets" / "food_background.jpg"

# Beautiful background
if BG_PATH.exists():
    image = base64.b64encode(BG_PATH.read_bytes()).decode()

    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background:
        linear-gradient(
            rgba(12, 35, 24, 0.80),
            rgba(12, 35, 24, 0.88)
        ),
        url("data:image/jpeg;base64,{image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    h1, h2, h3, p, label,
    [data-testid="stMarkdownContainer"] {{
        color: #fff8e8;
    }}

    .hero {{
        background: rgba(10, 35, 24, 0.85);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #e9bc65;
    }}

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {{
        background: #e9bc65;
        color: #173425;
        border-radius: 10px;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #102d21, #276348, #a06c36
        );
    }
    h1, h2, h3, p, label {
        color: #fff8e8;
    }
    </style>
    """, unsafe_allow_html=True)

    st.warning("Add assets/food_background.jpg for the background photo.")

# Header
st.markdown("""
<div class="hero">
    <h1>🍲 AI Recipe Chef</h1>
    <p>Enter your favourite food and discover a recipe!</p>
    <p>Fresh ideas • Your preferences • Easy cooking</p>
</div>
""", unsafe_allow_html=True)

# Load the local AI model
@st.cache_resource
def load_model():
    name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tokenizer, model

# User input
st.markdown("## 🥗 Create Your Recipe")

dish = st.text_input(
    "Enter food item",
    placeholder="Example: Chicken Biryani"
)

diet = st.selectbox(
    "Food preference",
    ["No special preference", "Vegetarian", "Vegan", "Non-vegetarian"]
)

spice = st.selectbox(
    "Spice level",
    ["Mild", "Medium", "Spicy"]
)

servings = st.selectbox(
    "Number of servings",
    [1, 2, 3, 4, 5, 6]
)

ingredients = st.text_input(
    "Available ingredients (optional)",
    placeholder="Rice, tomato, onion..."
)

# Generate recipe
if st.button("✨ Generate Recipe", use_container_width=True):

    if not dish.strip():
        st.error("Please enter a food item.")
    else:
        try:
            prompt = (
                f"Write a simple cooking recipe for {dish}. "
                f"Food preference: {diet}. "
                f"Spice level: {spice}. "
                f"Servings: {servings}. "
                f"Available ingredients: {ingredients or 'any suitable ingredients'}. "
                "Give ingredients with quantities and numbered cooking steps. "
                "Use simple English and realistic cooking instructions."
            )

            with st.spinner("👨‍🍳 Preparing your recipe..."):
                tokenizer, model = load_model()

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=220,
                        num_beams=2
                    )

                recipe = tokenizer.decode(
                    output[0],
                    skip_special_tokens=True
                )

            st.markdown("## 🍽️ Your Recipe")
            st.subheader(dish.title())

            # Display a matching food photo if available
            image_names = {
                "paneer butter masala": "paneer.jpg",
                "chicken biryani": "biryani.jpg",
                "biryani": "biryani.jpg",
                "masala dosa": "dosa.jpg",
                "dosa": "dosa.jpg",
                "veg fried rice": "fried_rice.jpg",
                "fried rice": "fried_rice.jpg"
            }

            food_folder = BASE_DIR / "assets" / "dishes"
            filename = image_names.get(dish.strip().lower())
            food_image = food_folder / filename if filename else None

            if food_image and food_image.exists():
                st.image(
                    str(food_image),
                    caption=dish.title(),
                    use_container_width=True
                )
            else:
                st.info(
                    "No photo saved for this dish yet. "
                    "You can add one in assets/dishes."
                )

            st.write(recipe)

            st.download_button(
                "📥 Download Recipe",
                data=f"{dish.title()}\n\n{recipe}",
                file_name="my_recipe.txt",
                mime="text/plain"
            )

            st.caption(
                "Check food safety, allergies, and cooking "
                "temperatures before preparing the recipe."
            )

        except Exception as error:
            st.error("Recipe generation failed. Check the error below.")
            st.exception(error)

st.markdown("---")
st.caption("Made with ❤️ using Python and Generative AI")
