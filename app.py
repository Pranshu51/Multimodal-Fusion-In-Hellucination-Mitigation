# -----------------------------------------------------------
# 🧠 AI Hallucination Reduction Demo using Grounded Prompting
# -----------------------------------------------------------
# This app demonstrates how grounded prompts help reduce
# hallucinations in Gemini compared to normal prompts.
# -----------------------------------------------------------

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# -----------------------------------------------------------
# 1️⃣ Load API Key
# -----------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------------------------------------
# 2️⃣ Define Helper Function to Call Gemini
# -----------------------------------------------------------
def generate_response(prompt, temperature=0.7):
    """
    Uses Gemini to generate a text response for a given prompt.
    Lower temperature → more factual, less creative output.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-1.5-pro"
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text

# -----------------------------------------------------------
# 3️⃣ Define Grounded Prompt Template
# -----------------------------------------------------------
def grounded_prompt(context, question):
    """
    Template that tells the model to use only the given context
    and avoid making assumptions.
    """
    return f"""
You are a careful factual assistant. 
Use ONLY the information provided below to answer the question.
If something is missing or uncertain, reply exactly with:
"Information not available in the given context."

Context:
{context}

Question: {question}

Instructions:
- Do NOT invent or assume any information.
- Base your answer only on what is in the context.
- Be clear, short, and factual.
"""

# -----------------------------------------------------------
# 4️⃣ Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="AI Hallucination Reduction Demo", layout="wide")

st.title("🧠 AI Hallucination Reduction Demo")
st.write("### Using **Google Gemini** with **Grounded Prompting** to reduce AI hallucination.")
st.markdown("---")

st.markdown("""
#### 📘 Instructions:
1. Enter a factual context or passage in the left box.
2. Type a question below it.
3. Click **Run Demo** to see:
   - A normal Gemini answer (which may hallucinate).
   - A grounded Gemini answer (hallucination-reduced).
""")

# -----------------------------------------------------------
# 5️⃣ Input Fields
# -----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    context = st.text_area("Enter Context (text passage, article, or data):", 
        height=200,
        placeholder="Example:\nThe Eiffel Tower is located in Paris, France. It was completed in 1889 as part of the World Fair. The tower is 330 meters tall.")

with col2:
    question = st.text_area("Enter Question:", 
        height=200,
        placeholder="Example: When was the Eiffel Tower moved to London?")

run_button = st.button("🚀 Run Demo")

# -----------------------------------------------------------
# 6️⃣ Process and Display Outputs
# -----------------------------------------------------------
if run_button:
    if not context or not question:
        st.warning("⚠️ Please enter both a context and a question.")
    else:
        with st.spinner("Generating responses..."):
            # Normal (uncontrolled) response
            normal_prompt = f"Question: {question}\nAnswer in detail."
            normal_answer = generate_response(normal_prompt, temperature=1.0)

            # Grounded (controlled) response
            grounded_text = grounded_prompt(context, question)
            grounded_answer = generate_response(grounded_text, temperature=0.2)

        # Display Results
        st.markdown("---")
        st.subheader("🧩 Results Comparison")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### ❌ Normal Gemini Response (May Hallucinate)")
            st.info(normal_answer)
        with colB:
            st.markdown("### ✅ Grounded Gemini Response (Reduced Hallucination)")
            st.success(grounded_answer)

        # -------------------------------------------------------
        # Optional: Simple Hallucination Indicator
        # -------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Hallucination Check")

        if "Information not available" in grounded_answer:
            st.warning("⚠️ The grounded model detected missing or unverifiable information and refused to guess — hallucination avoided.")
        elif grounded_answer.lower() in normal_answer.lower():
            st.success("✅ Both answers match — no hallucination detected.")
        else:
            st.info("ℹ️ Grounded response differs — likely a hallucination was prevented.")

        st.markdown("---")
        st.caption("Model: Gemini 2.5 Flash | Technique: Grounded Prompting | Developed by Pranshu’s Team 🧠")

