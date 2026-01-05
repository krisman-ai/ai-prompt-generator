import streamlit as st
from prompt_engine import PromptInputs, build_base_prompt, refine_with_ai

st.set_page_config(page_title="AI Prompt Generator", page_icon="🧠", layout="wide")

st.title("🧠 AI Prompt Generator (Portfolio Tool)")
st.caption("Bikin prompt berkualitas tinggi untuk marketer, designer, dan ChatGPT user — pakai Google Gemini API.")

# ===== Sidebar Inputs =====
st.sidebar.header("Input")

role = st.sidebar.selectbox("Role", ["ChatGPT User", "Marketer", "Designer", "Content Writer", "Product Manager"])
task = st.sidebar.selectbox("Task", ["Write Caption", "Write Script", "Write Landing Page", "Write Email", "Generate Ideas", "Write Brief"])
goal = st.sidebar.text_input("Goal (tujuan)", "Meningkatkan engagement & konversi")
audience = st.sidebar.text_input("Audience", "Pemula 18–30, Indonesia")
tone = st.sidebar.selectbox("Tone", ["Friendly", "Professional", "Casual", "Bold", "Luxury", "Funny"])
language = st.sidebar.selectbox("Language", ["Indonesian", "English"])

context = st.sidebar.text_area("Context (opsional)", "")
constraints = st.sidebar.text_area("Constraints (opsional)", "Singkat, jelas, ada CTA")
output_format = st.sidebar.text_area("Output format (opsional)", "Bullet points + 3 variasi")

inputs = PromptInputs(
    role=role,
    task=task,
    goal=goal,
    audience=audience,
    tone=tone,
    context=context,
    constraints=constraints,
    output_format=output_format,
    language=language,
)

col1, col2 = st.columns(2)

# ===== Left: Base Prompt =====
with col1:
    st.subheader("1) Base Prompt (template)")
    if st.button("🧩 Generate Base Prompt"):
        st.session_state["base_prompt"] = build_base_prompt(inputs)

    base_prompt = st.text_area(
        "Base prompt",
        value=st.session_state.get("base_prompt", ""),
        height=320,
        placeholder="Klik tombol Generate Base Prompt atau tulis manual di sini..."
    )

    if base_prompt:
        st.download_button(
            "⬇️ Download Base Prompt (.txt)",
            data=base_prompt,
            file_name="base_prompt.txt",
            mime="text/plain",
        )

# ===== Right: Gemini Refine =====
with col2:
    st.subheader("2) AI-Refined Prompt (Gemini API)")
    st.caption('Saat deploy, isi GOOGLE_API_KEY di Streamlit Secrets.')

    model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

    if st.button("🤖 Refine with AI"):
        try:
            if not base_prompt.strip():
                st.warning("Base prompt masih kosong. Buat dulu base prompt.")
            else:
                refined = refine_with_ai(base_prompt, model_name=model_name)
                st.session_state["refined_prompt"] = refined
        except Exception as e:
            st.error(f"Gagal refine: {e}")

    refined_prompt = st.text_area(
        "Refined prompt",
        value=st.session_state.get("refined_prompt", ""),
        height=320,
        placeholder="Klik Refine with AI untuk menghasilkan prompt final..."
    )

    if refined_prompt:
        st.download_button(
            "⬇️ Download Refined Prompt (.txt)",
            data=refined_prompt,
            file_name="refined_prompt.txt",
            mime="text/plain",
        )
