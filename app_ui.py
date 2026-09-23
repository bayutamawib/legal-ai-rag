import os
import requests
import streamlit as st

# Konfigurasi URL API (Gunakan nama container di Docker network atau localhost untuk lokal)
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/query")

# Streamlit page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Main Header
st.title("⚖️ Legal AI Assistant")
st.markdown(
    "Sistem Tanya-Jawab Hukum Indonesia berbasis **Retrieval-Augmented Generation (RAG)**."
)
st.markdown("---")

# Configuration & Parameter Sidebar
with st.sidebar:
    st.header("⚙️ Retrieval Setting")
    top_k = st.slider(
        "Kandidat Awal (Top-K)",
        min_value=3,
        max_value=20,
        value=10,
        help="Jumlah kandidat dokumen yang diambil dari Qdrant."
    )
    top_n = st.slider(
        "Hasil Re-ranking (Top-N)",
        min_value=1,
        max_value=10,
        value=3,
        help="Jumlah pasal paling relevan yang dikirim ke LLM setelah dinilai ulang."
    )
    use_cache = st.checkbox("Gunakan Redis Caching", value=True)

    st.markdown("---")
    st.header("📊 Arsitektur Sistem")
    st.markdown("- **Backend**: FastAPI + Redis Caching")
    st.markdown("- **Vector Store**: Qdrant")
    st.markdown("- **Embedding**: `BAAI/bge-m3`")
    st.markdown("- **Reranker**: `BAAI/bge-reranker-base`")
    st.markdown("- **LLM**: Groq Llama 3.3 70B")

# Input Question Form
query = st.text_input(
    "Tanyakan tentang regulasi/hukum di Indonesia:",
    placeholder="Contoh: Apa saja kewajiban Penyelenggara Sistem Elektronik?",
)

if st.button("Cari & Analisis", type="primary") and query:
    with st.spinner("Menghubungi API & menyusun analisis hukum..."):
        payload = {
            "query": query,
            "top_k": top_k,
            "top_n": top_n,
            "use_cache": use_cache
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Badge Status Cache
                if result.get("cached"):
                    st.success("⚡ Hasil diambil dari Redis Cache")
                else:
                    st.info("🧠 Hasil dihitung langsung via RAG Pipeline")

                # Main Answer Layout
                st.subheader("💡 Legal Analysis")
                st.markdown(result["answer"])

                st.markdown("---")

                # Citation Layout and Article Reference
                st.subheader("📜 Legal Base & Reference")
                sources = result.get("sources", [])
                
                if sources:
                    for idx, src in enumerate(sources, 1):
                        col1, col2 = st.columns([1, 3])

                        doc_id = src.get("document_id", "")
                        pasal_title = src.get("pasal", "")
                        
                        # Gabungkan nama dokumen + Pasal (misal: PP Nomor 71 Tahun 2019 Pasal 24)
                        label_lengkap = f"{doc_id} {pasal_title}".strip() if doc_id else pasal_title

                        with col1:
                            st.caption(f"**SUMBER #{idx}**")
                            st.markdown(f"### 📜 {label_lengkap}")
                            st.caption(f"🎯 Skor: `{src['score']:.4f}`")
                    
                        with col2:
                            with st.expander("Lihat Ringkasan Teks", expanded=True):
                                st.caption(f"**{src.get('bab', '')}**")
                                st.write(src['content_preview'])
                            
                        st.divider() # Garis pembatas antar pasal
                else: # <-- INDENTASI SEJAJAR DENGAN 'if sources:'
                    st.info("Tidak ada referensi pasal yang terdeteksi.")
            else:
                st.error(f"Gagal memproses permintaan (HTTP {response.status_code}): {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal terhubung ke API Server di `{API_URL}`: {e}")