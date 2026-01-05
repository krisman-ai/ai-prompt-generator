import streamlit as st
from prompt_engine import PromptInputs, build_base_prompt, refine_prompt_with_openai

st.set_page_config(page_title="AI Prompt Generator", page_icon="🧠", layout="wide")

st.title("🧠 AI Prompt Generator (Portfolio Tool)")
st.caption("Bikin prompt berkualitas tinggi untuk marketer, designer, dan ChatGPT user — pakai OpenAI API.")

with st.sidebar:
    st.header("Input")

    role = st.selectbox("Role", ["Marketer", "Designer", "ChatGPT User", "Student", "Business"], index=2)

    task = st.selectbox(
        "Task",
        ["Write Caption", "Generate Content Ideas", "Landing Page Copy", "Design Brief", "Email", "Prompt Improvement"],
        index=0,
    )

    goal = st.text_input("Goal (tujuan)", "Meningkatkan engagement & konversi.")
    audience = st.text_input("Audience", "Pemula 18–30, Indonesia")
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Fun", "Luxury", "Direct"], index=0)
    language = st.selectbox("Language", ["Indonesian", "English"], index=0)

    output_format = st.text_input("Output format", "Bullet points + contoh + CTA")
    constraints = st.text_area("Constraints (opsional)", "Max 150 kata per versi. Hindari istilah teknis.")
    context = st.text_area("Context", "Brand: ...\nProduk: ...\nPlatform: ...\nKeunggulan: ...")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Base Prompt (template)")
    if st.button("🧩 Generate Base Prompt"):
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
        st.session_state["base_prompt"] = build_base_prompt(p)

    base_prompt = st.text_area(
        "Base prompt",
        value=st.session_state.get("base_prompt", ""),
        height=320,
        placeholder="Klik tombol Generate Base Prompt atau tulis manual di sini...",
    )

    st.download_button(
        "⬇️ Download Base Prompt (.txt)",
        data=base_prompt.encode("utf-8"),
        file_name="base_prompt.txt",
        mime="text/plain",
        disabled=not base_prompt.strip(),
    )

with col2:
    st.subheader("2) AI-Refined Prompt (OpenAI API)")
    st.caption("Nanti saat deploy, isi OPENAI_API_KEY di Streamlit Secrets.")

    model = st.text_input("Model", value="gpt-4.1-mini")

    if st.button("🤖 Refine with AI"):
        if not base_prompt.strip():
            st.error("Base prompt masih kosong. Klik 'Generate Base Prompt' dulu.")
        else:
            api_key = None
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", None)
            except Exception:
                api_key = None

            try:
                refined_md = refine_prompt_with_openai(base_prompt, api_key=api_key, model=model)
                st.session_state["refined_md"] = refined_md
            except Exception as e:
                st.error(f"Gagal refine: {e}")

    refined = st.session_state.get("refined_md", "")
    if refined:
        st.markdown(refined)
        st.download_button(
            "⬇️ Download Refined Prompt (.md)",
            data=refined.encode("utf-8"),
            file_name="refined_prompt.md",
            mime="text/markdown",
        )
    else:
        st.info("Klik **Refine with AI** untuk menghasilkan prompt final.")

st.divider()
st.subheader("Cara pakai cepat")
st.write("1) Isi input → 2) Generate Base Prompt → 3) Refine with AI → 4) Copy hasil prompt.")
