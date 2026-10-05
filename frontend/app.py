"""
Streamlit Frontend — Credit Policy & Circular Assistant UI.

Features:
  - Query input with example questions
  - Answer display with highlighted citations
  - Expandable retrieved context panels
  - Strategy selector for comparison
  - Evaluation metrics dashboard
"""

import html
import json
import time
from pathlib import Path

import httpx
import streamlit as st

# ── Page Configuration ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Policy & Circular Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ───────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #e2e2e2, #a8d8ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #a8b2c1;
        margin-top: 0.5rem;
        font-size: 0.95rem;
    }

    /* These cards use light backgrounds, so text colours are set explicitly —
       otherwise the app's theme (e.g. dark mode) renders white text on them. */
    .answer-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        line-height: 1.7;
        color: #1e293b;
    }
    .answer-card strong { color: #0f172a; }

    .citation-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }

    .citation-unverified {
        background: #fef3c7;
        color: #92400e;
    }

    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-card h3 {
        color: #0369a1;
        font-size: 1.8rem;
        margin: 0;
    }
    .metric-card p {
        color: #475569;
        font-size: 0.85rem;
        margin: 0.25rem 0 0;
    }

    .context-panel {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1e293b;
    }
    .context-panel em { color: #334155; }

    .abstention-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid #ef4444;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        color: #7f1d1d;
    }

    .timing-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 8px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Configuration ────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"

EXAMPLE_QUESTIONS = [
    "What is the minimum CRAR requirement for NBFC-ML under RBI directions?",
    "What are the KYC requirements for opening accounts at NBFCs?",
    "Explain the fair practices code requirements for loan recovery.",
    "What are the digital lending guidelines for NBFC-P2P platforms?",
    "What is the maximum exposure limit for a single borrower for NBFC-ND-SI?",
    "Compare the NPA classification norms for NBFC-ML and banks.",
    "What are the outsourcing guidelines for NBFCs regarding data security?",
    "What is the liquidity coverage ratio requirement for NBFC-ML?",
]


# ── Helper Functions ─────────────────────────────────────────────────────

def query_api(question: str, strategy: str = "structural", top_k: int = 5) -> dict:
    """Send a query to the FastAPI backend."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/query",
            json={
                "question": question,
                "strategy": strategy,
                "top_k": top_k,
                "include_context": True,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error("❌ Cannot connect to API server. Make sure it's running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None


def get_health() -> dict | None:
    """Check API health."""
    try:
        response = httpx.get(f"{API_BASE_URL}/api/v1/health", timeout=5.0)
        return response.json()
    except Exception:
        return None


def format_citation(citation: dict) -> str:
    """Format a citation as a readable badge."""
    parts = [citation.get("document_title", "Unknown")]
    if citation.get("section"):
        parts.append(citation["section"])
    if citation.get("page"):
        parts.append(f"Page {citation['page']}")

    text = ", ".join(parts)
    css_class = "citation-badge" if citation.get("verified", True) else "citation-badge citation-unverified"
    icon = "✓" if citation.get("verified", True) else "⚠️"
    return f'<span class="{css_class}">{icon} {text}</span>'


# ── Main App ─────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Credit Policy & Circular Assistant</h1>
        <p>RAG-powered Q&A over RBI Master Directions, NBFC Credit Policies & Product Circulars</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        strategy = st.selectbox(
            "Chunking Strategy",
            ["structural", "recursive_overlap", "fixed_512"],
            index=0,
            help="Select the chunking strategy for retrieval comparison",
        )

        top_k = st.slider("Results (Top K)", 3, 15, 5)

        st.divider()

        # Health check
        health = get_health()
        if health:
            status_color = "🟢" if health.get("status") == "healthy" else "🟡"
            st.markdown(f"**Status:** {status_color} {health.get('status', 'unknown')}")
            st.markdown(f"**Qdrant:** {'✅' if health.get('qdrant_connected') else '❌'}")
            st.markdown(f"**Documents:** {health.get('documents_indexed', 0):,}")
            st.markdown(f"**Embedder:** `{health.get('embedding_model', 'N/A')}`")
        else:
            st.markdown("**Status:** 🔴 API Offline")
            st.caption("Start the API with `python -m src.api.main`")

        st.divider()
        st.markdown("### 📝 Example Questions")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            label = q if len(q) <= 60 else q[:57] + "..."
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                # Write to the text area's own widget key. This runs before the
                # text area is created, which is the only point Streamlit allows
                # a widget's state to be set.
                st.session_state["query_box"] = q
                st.session_state["run_query"] = True

    # ── Main Content ─────────────────────────────────────────────────

    # Query Input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area(
            "Ask a regulatory question:",
            height=80,
            placeholder="e.g., What is the minimum CRAR requirement for NBFC-ML?",
            key="query_box",
        )
    with col2:
        st.write("")  # spacing
        st.write("")
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Clicking an example question triggers the search on this rerun
    auto_run = st.session_state.pop("run_query", False)

    # Process Query
    if (ask_button or auto_run) and query.strip():
        with st.spinner("🔄 Searching policy documents..."):
            result = query_api(query.strip(), strategy, top_k)

        if result:
            # ── Timing badges ────────────────────────────────────────
            timing = result.get("timing", {})
            timing_html = "".join([
                f'<span class="timing-badge">🔍 Retrieval: {timing.get("retrieval_ms", 0):.0f}ms</span>',
                f'<span class="timing-badge">📊 Rerank: {timing.get("rerank_ms", 0):.0f}ms</span>',
                f'<span class="timing-badge">🤖 Generation: {timing.get("generation_ms", 0):.0f}ms</span>',
                f'<span class="timing-badge">⏱️ Total: {timing.get("total_ms", 0):.0f}ms</span>',
            ])
            st.markdown(timing_html, unsafe_allow_html=True)

            # ── Answer Display ───────────────────────────────────────
            abstention = result.get("abstention", {})

            if abstention.get("abstained"):
                st.markdown(f"""
                <div class="abstention-card">
                    <strong>⚠️ Abstention</strong><br/>
                    {result.get('answer', '')}
                    <br/><br/>
                    <small>Reason: {abstention.get('reason', '')}<br/>
                    Max rerank score: {abstention.get('max_score', 0):.4f} 
                    (threshold: {abstention.get('threshold', 0):.4f})</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Escape model/document text before injecting into HTML; keep line breaks
                answer_html = html.escape(result.get("answer", "No answer generated.")).replace("\n", "<br/>")
                st.markdown(
                    f'<div class="answer-card">{answer_html}</div>',
                    unsafe_allow_html=True,
                )

            # ── Citations ────────────────────────────────────────────
            citations = result.get("citations", [])
            if citations:
                st.markdown("#### 📚 Citations")
                citations_html = " ".join(format_citation(c) for c in citations)
                st.markdown(citations_html, unsafe_allow_html=True)

                verified = result.get("verified_citations", 0)
                unverified = result.get("unverified_citations", 0)
                st.caption(f"✅ {verified} verified  |  ⚠️ {unverified} unverified")

            # ── Token Usage ──────────────────────────────────────────
            token_usage = result.get("token_usage", {})
            if token_usage.get("total", 0) > 0:
                st.caption(
                    f"Tokens — Prompt: {token_usage.get('prompt', 0):,} | "
                    f"Completion: {token_usage.get('completion', 0):,} | "
                    f"Total: {token_usage.get('total', 0):,}"
                )

            # ── Retrieved Context ────────────────────────────────────
            context_chunks = result.get("context_chunks", [])
            if context_chunks:
                with st.expander(f"📄 Retrieved Context ({len(context_chunks)} chunks)", expanded=False):
                    for i, chunk in enumerate(context_chunks):
                        section = " > ".join(chunk.get("section_hierarchy", [])) or "N/A"
                        st.markdown(f"""
                        <div class="context-panel">
                            <strong>Chunk {i+1}</strong> — 
                            <em>{chunk.get('document_title', 'Unknown')}</em><br/>
                            📑 Section: {section} | 
                            📄 Page: {chunk.get('page_number', 'N/A')} | 
                            Score: {chunk.get('rerank_score', 0):.4f}<br/>
                            <hr style="margin: 0.5rem 0"/>
                            {chunk.get('text', '')[:400]}...
                        </div>
                        """, unsafe_allow_html=True)

    # ── Strategy Comparison Tab ──────────────────────────────────────
    st.divider()

    tab1, tab2 = st.tabs(["📊 Strategy Comparison", "📈 Evaluation Metrics"])

    with tab1:
        st.markdown("### Chunking Strategy Benchmark Results")
        st.caption(
            "Compare retrieval quality across three chunking strategies. "
            "Run the benchmark to populate this table."
        )

        # Try loading comparison results
        try:
            comparison_path = Path("data/processed/benchmarks/strategy_comparison.json")
            if comparison_path.exists():
                with open(comparison_path) as f:
                    comparison_data = json.load(f)

                # Build comparison DataFrame
                import pandas as pd
                rows = []
                for m in comparison_data:
                    rows.append({
                        "Strategy": m.get("strategy", ""),
                        "Recall@5": f"{m.get('retrieval', {}).get('recall_at_5', 0):.4f}",
                        "MRR": f"{m.get('retrieval', {}).get('mrr', 0):.4f}",
                        "Faithfulness": f"{m.get('faithfulness', {}).get('mean', 0):.4f}",
                        "Abstention P": f"{m.get('abstention', {}).get('abstention_precision', 0):.2f}",
                        "Abstention R": f"{m.get('abstention', {}).get('abstention_recall', 0):.2f}",
                        "p50 Latency": f"{m.get('latency', {}).get('p50', 0):.0f}ms",
                        "p95 Latency": f"{m.get('latency', {}).get('p95', 0):.0f}ms",
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No benchmark results yet. Run `python scripts/run_evaluation.py` to generate.")
        except Exception as e:
            st.warning(f"Could not load comparison data: {e}")

    with tab2:
        st.markdown("### Evaluation Metrics Detail")
        try:
            metrics_resp = httpx.get(f"{API_BASE_URL}/api/v1/metrics", timeout=5.0)
            if metrics_resp.status_code == 200:
                metrics_data = metrics_resp.json()
                strategies = metrics_data.get("strategies", [])
                if strategies:
                    for strat in strategies:
                        with st.expander(f"📊 {strat.get('strategy', 'Unknown')}", expanded=False):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Recall@5", f"{strat.get('retrieval', {}).get('recall_at_5', 0):.3f}")
                            with col2:
                                st.metric("MRR", f"{strat.get('retrieval', {}).get('mrr', 0):.3f}")
                            with col3:
                                st.metric("Faithfulness", f"{strat.get('faithfulness', {}).get('mean', 0):.3f}")
                            with col4:
                                st.metric("p50 Latency", f"{strat.get('latency', {}).get('p50', 0):.0f}ms")
                else:
                    st.info("No evaluation metrics available yet.")
            else:
                st.info("API not returning metrics. Run evaluation first.")
        except Exception:
            st.info("Connect to the API to view evaluation metrics.")


if __name__ == "__main__":
    main()
