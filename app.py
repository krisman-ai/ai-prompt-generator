import streamlit as st

from prompt_engine import PromptInputs, build_base_prompt, refine_with_ai


st.set_page_config(page_title="AI Prompt Generator", page_icon="🧠", layout="wide")

st.title("🧠 AI Prompt Generator (Portfolio Tool)")
st.caption("Bikin prompt berkualitas tinggi untuk marketer, designer, dan ChatGPT user — pakai Google Gemini API.")


# --- Sidebar inputs
st.sidebar.header("Input")

role = st.sidebar.selectbox("Role", ["ChatGPT User", "Marketer", "Designer", "Content Writer"], index=0)
task = st.sidebar.selectbox("Task", ["Write Caption", "Write Landing Page", "Write Email", "Create Design Brief"], index=0)
goal = st.sidebar.text_input("Goal (tujuan)", value="Meningkatkan engagement & konversi")
audience = st.sidebar.text_input("Audience", value="Pemula 18–30, Indonesia")
tone = st.sidebar.selectbox("Tone", ["Friendly", "Professional", "Bold", "Luxury"], index=0)
language = st.sidebar.selectbox("Language", ["Indonesian", "English"], index=0)

context = st.sidebar.text_area("Context (opsional)", value="")
constraints = st.sidebar.text_area("Constraints (opsional)", value="")
output_format = st.sidebar.text_area("Output format (opsional)", value="")

# --- Model selection
st.sidebar.divider()
st.sidebar.subheader("AI Settings (Gemini)")
selected_model = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)

# --- Build base prompt
p = PromptInputs(
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

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("1) Base Prompt (template)")

    if st.button("🧩 Generate Base Prompt"):
        st.session_state["base_prompt"] = build_base_prompt(p)

    base_prompt = st.session_state.get("base_prompt", "")
    base_prompt = st.text_area(
        "Base prompt",
        value=base_prompt,
        height=320,
        placeholder="Klik tombol Generate Base Prompt atau tulis manual di sini..."
    )
    st.session_state["base_prompt"] = base_prompt

    st.download_button(
        "⬇️ Download Base Prompt (.txt)",
        data=base_prompt or "",
        file_name="base_prompt.txt",
        mime="text/plain"
    )

with col2:
    st.subheader("2) AI-Refined Prompt (Gemini API)")
    st.caption("Saat deploy, isi GOOGLE_API_KEY di Streamlit Secrets.")

    st.write(f"**Model:** `{selected_model}`")

    if st.button("🤖 Refine with AI"):
        if not base_prompt.strip():
            st.error("Base prompt masih kosong. Klik Generate Base Prompt dulu.")
        else:
            try:
                refined = refine_with_ai(
                    base_prompt=base_prompt,
                    model=selected_model,      # ✅ FIX: model (bukan model_name)
                    temperature=temperature,
                )
                st.session_state["refined_prompt"] = refined
                st.success("Berhasil refine! ✅")
            except Exception as e:
                st.error(f"Gagal refine: {e}")

    refined_prompt = st.session_state.get("refined_prompt", "")
    st.text_area(
        "Refined prompt",
        value=refined_prompt,
        height=320,
        placeholder="Klik Refine with AI untuk menghasilkan prompt final..."
    )

    st.download_button(
        "⬇️ Download Refined Prompt (.txt)",
        data=refined_prompt or "",
        file_name="refined_prompt.txt",
        mime="text/plain"
    )
