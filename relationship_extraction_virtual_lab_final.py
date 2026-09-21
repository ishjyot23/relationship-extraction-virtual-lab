import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import re
import json
import random
from pathlib import Path

# Optional document readers
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

# Optional ML relation-extraction model
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    AutoTokenizer = AutoModelForSeq2SeqLM = None


st.set_page_config(page_title="Relationship Extraction | Virtual Lab", layout="wide")

TITLE = "Relationship Extraction from Text"
AIM = (
    "To extract semantic relationships between identified entities from unstructured "
    "text and represent them as Entity–Relationship triples suitable for graph construction."
)
OUTCOME = "Entity–relationship triples suitable for graph construction."

DEFAULT = (
    "Sundar Pichai is the CEO of Google. Google is headquartered in Mountain View. "
    "Google developed Gemini. Sundar Pichai studied at Stanford University."
)

EXAMPLES = {
    "Google example": DEFAULT,
    "Microsoft–LinkedIn example": (
        "Microsoft acquired LinkedIn. LinkedIn is headquartered in Sunnyvale."
    ),
    "SpaceX example": "Elon Musk founded SpaceX. SpaceX developed Starship.",
}

PATTERNS = {
    "is CEO of": ["is the ceo of", "is ceo of", "chief executive officer of", "ceo of"],
    "developed": ["developed", "created", "built", "introduced"],
    "founded": ["founded", "co-founded", "cofounded", "established", "started"],
    "headquartered in": ["headquartered in", "based in", "located in"],
    "studied at": ["studied at", "graduated from", "attended"],
    "acquired": ["acquired", "bought", "purchased"],
    "works at": ["works at", "works for", "employee of", "joined"],
    "partnered with": ["partnered with", "collaborated with", "worked with"],
}

QUIZ_BANK_PATH = Path(__file__).resolve().parent / "quiz_bank.json"


def load_quiz_bank():
    try:
        with QUIZ_BANK_PATH.open("r", encoding="utf-8") as f:
            bank = json.load(f)
    except FileNotFoundError:
        st.error("quiz_bank.json is missing. Keep it in the same Git repository/folder as this .py file.")
        st.stop()
    except json.JSONDecodeError as exc:
        st.error(f"quiz_bank.json is invalid: {exc}")
        st.stop()

    if len(bank) < 10:
        st.error("The quiz bank must contain at least 10 questions.")
        st.stop()

    return bank


QUIZ_BANK = load_quiz_bank()
QUIZ_SIZE = 10

BOOK_REFERENCES = [
    (
        "Jurafsky, D. & Martin, J. H. Speech and Language Processing, 3rd ed. Online manuscript, Stanford University.",
        "https://web.stanford.edu/~jurafsky/slp3/",
    ),
    (
        "Tunstall, L., von Werra, L. & Wolf, T. Natural Language Processing with Transformers. O'Reilly Media, 2022.",
        "https://www.oreilly.com/library/view/natural-language-processing/9781098136789/",
    ),
    (
        "Jain, S. M. Introduction to Transformers for NLP. Apress, 2022.",
        "https://link.springer.com/book/10.1007/978-1-4842-8844-3",
    ),
]

PAPER_REFERENCES = [
    (
        "Huguet Cabot, P.-L. & Navigli, R. (2021). REBEL: Relation Extraction By End-to-end Language generation. Findings of EMNLP 2021, 2370–2381.",
        "https://aclanthology.org/2021.findings-emnlp.204/",
    ),
    (
        "Christopoulou, F., Miwa, M. & Ananiadou, S. (2021). Distantly Supervised Relation Extraction with Sentence Reconstruction. EACL 2021.",
        "https://aclanthology.org/2021.eacl-main.11/",
    ),
    (
        "Zeng, D., Liu, K., Lai, S., Zhou, G. & Zhao, J. (2014). Relation Classification via Convolutional Deep Neural Network. COLING 2014.",
        "https://aclanthology.org/C14-1220/",
    ),
]

BLOG_REFERENCES = [
    (
        "spaCy Usage Documentation — spaCy 101 / Named Entity Recognition.",
        "https://spacy.io/usage/spacy-101/",
    ),
    (
        "spaCy API Documentation — EntityRuler and linguistic processing components.",
        "https://spacy.io/api/",
    ),
    (
        "Streamlit Documentation — st.file_uploader.",
        "https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader",
    ),
    (
        "Hugging Face Transformers Documentation — Text-to-Text Generation.",
        "https://huggingface.co/docs/transformers/tasks/text2text",
    ),
]


@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model missing. Run: python -m spacy download en_core_web_sm")
        st.stop()


@st.cache_resource(show_spinner="Loading REBEL transformer model (first run may take a while)...")
def load_rebel_model():
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError(
            "Transformers is not installed. Add transformers, torch and sentencepiece to requirements.txt."
        )

    model_name = "Babelscape/rebel-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


nlp = load_spacy_model()


def norm(x):
    return " ".join(x.lower().split())


def relation(e1, e2, sent):
    low = norm(sent.text)
    a = min(e1.end_char, e2.end_char)
    b = max(e1.start_char, e2.start_char)
    between = norm(sent.text[a:b])

    hits = []
    for r, phrases in PATTERNS.items():
        for p in phrases:
            if p in between or p in low:
                hits.append((len(p), r))

    if hits:
        return sorted(hits, reverse=True)[0][1]

    lo, hi = min(e1.start, e2.start), max(e1.start, e2.start)
    verbs = [t.lemma_ for t in sent if lo <= t.i <= hi and t.pos_ == "VERB"]
    return verbs[0] if verbs else None


def extract_spacy(text):
    """spaCy NER + lightweight linguistic relation fallback."""
    doc = nlp(text)
    ents, seen = [], set()
    triples, tseen = [], set()
    candidate_pairs = []

    for e in doc.ents:
        key = (e.text.lower(), e.label_)
        if key not in seen:
            seen.add(key)
            ents.append({"Entity": e.text, "Type": e.label_})

    for sent in doc.sents:
        sentence_entities = list(sent.ents)
        for i in range(len(sentence_entities)):
            for j in range(i + 1, len(sentence_entities)):
                e1, e2 = sentence_entities[i], sentence_entities[j]
                candidate_pairs.append(
                    {"Entity 1": e1.text, "Entity 2": e2.text, "Sentence": sent.text.strip()}
                )

                r = relation(e1, e2, sent)
                if not r:
                    continue

                sub, obj = (e1, e2) if e1.start < e2.start else (e2, e1)
                key = (sub.text.lower(), r.lower(), obj.text.lower())
                if key not in tseen:
                    tseen.add(key)
                    triples.append(
                        {
                            "Subject": sub.text,
                            "Relation": r,
                            "Object": obj.text,
                            "Source Sentence": sent.text.strip(),
                        }
                    )

    return ents, candidate_pairs, triples


def parse_rebel_output(text):
    """Parse REBEL: <triplet> subject <subj> relation <obj> object."""
    text = text.replace("<s>", "").replace("</s>", "").strip()
    pattern = re.compile(
        r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=\s*<triplet>|$)",
        re.IGNORECASE,
    )

    triples = []
    for subject, relation_text, obj in pattern.findall(text):
        subject, relation_text, obj = subject.strip(), relation_text.strip(), obj.strip()
        if subject and relation_text and obj:
            triples.append(
                {
                    "Subject": subject,
                    "Relation": relation_text,
                    "Object": obj,
                    "Source Sentence": "",
                }
            )
    return triples


def extract_rebel(text):
    """General ML relation extraction using pretrained REBEL."""
    tokenizer, model = load_rebel_model()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        sentences = [text[:4000]]

    all_triples, seen = [], set()
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=3,
            num_return_sequences=1,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
        for t in parse_rebel_output(decoded):
            key = (t["Subject"].lower(), t["Relation"].lower(), t["Object"].lower())
            if key not in seen:
                seen.add(key)
                t["Source Sentence"] = sent
                all_triples.append(t)

    ents, seen_entities = [], set()
    for e in doc.ents:
        key = (e.text.lower(), e.label_)
        if key not in seen_entities:
            seen_entities.add(key)
            ents.append({"Entity": e.text, "Type": e.label_})

    existing = {e["Entity"].lower() for e in ents}
    for t in all_triples:
        for name in (t["Subject"], t["Object"]):
            if name.lower() not in existing:
                ents.append({"Entity": name, "Type": "ML-identified entity"})
                existing.add(name.lower())

    candidate_pairs = []
    for sent in sentences:
        local_ents = [e.text for e in nlp(sent).ents]
        for i in range(len(local_ents)):
            for j in range(i + 1, len(local_ents)):
                candidate_pairs.append(
                    {"Entity 1": local_ents[i], "Entity 2": local_ents[j], "Sentence": sent}
                )

    return ents, candidate_pairs, all_triples


def extract(text, method):
    return extract_rebel(text) if method == "ML model (REBEL)" else extract_spacy(text)


def graph(triples, highlighted_nodes=None, highlighted_edges=None, title="Extracted Entity–Relationship Graph", standalone_nodes=None):
    """Build graph with optional node/edge highlighting. Explicit styling avoids duplicate Streamlit chart IDs."""
    highlighted_nodes = set(highlighted_nodes or [])
    highlighted_edges = set(highlighted_edges or [])

    nodes = []
    for t in triples:
        for n in (t["Subject"], t["Object"]):
            if n not in nodes:
                nodes.append(n)

    if standalone_nodes:
        for n in standalone_nodes:
            if n not in nodes:
                nodes.append(n)

    if not nodes:
        fig = go.Figure()
        fig.update_layout(height=360, title=title)
        return fig

    # Spread nodes over a simple two-row layout.
    pos = {n: (i * 2.8, 0 if i % 2 == 0 else 0.9) for i, n in enumerate(nodes)}
    fig = go.Figure()

    for t in triples:
        x1, y1 = pos[t["Subject"]]
        x2, y2 = pos[t["Object"]]
        edge_key = (t["Subject"], t["Relation"], t["Object"])
        active = edge_key in highlighted_edges
        width = 6 if active else 2
        fig.add_trace(
            go.Scatter(
                x=[x1, x2], y=[y1, y2], mode="lines",
                line=dict(width=width), showlegend=False, hoverinfo="none"
            )
        )
        fig.add_annotation(
            x=(x1 + x2) / 2, y=(y1 + y2) / 2 + 0.18,
            text=f"<b>{t['Relation']}</b>" if active else t["Relation"],
            showarrow=False,
        )
        fig.add_annotation(
            x=x2, y=y2, ax=x1, ay=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3,
            arrowwidth=4 if active else 2,
        )

    node_sizes = [54 if n in highlighted_nodes else 42 for n in nodes]
    # Plotly accepts a list of colors; highlighted nodes are visually distinct.
    node_colors = ["#F4A261" if n in highlighted_nodes else "#D9EAF7" for n in nodes]
    node_lines = ["#D35400" if n in highlighted_nodes else "#4A5568" for n in nodes]

    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            text=nodes,
            textposition="bottom center",
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=3, color=node_lines)),
            showlegend=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=480,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


def application_graph():
    applications = [
        "Knowledge Graphs", "Search & QA", "Recommendation",
        "Fraud / Compliance", "Biomedical NLP", "Social Media Analysis",
    ]
    fig = go.Figure()
    for i in range(1, len(applications) + 1):
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, i - 1], mode="lines", line=dict(width=2), showlegend=False, hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=[0] + [1] * len(applications),
            y=[0] + list(range(len(applications))),
            mode="markers+text",
            text=["Relationship\nExtraction"] + applications,
            textposition=["middle left"] + ["middle right"] * len(applications),
            marker=dict(size=[48] + [34] * len(applications)),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Real-World Applications of Relationship Extraction",
        height=430,
        xaxis=dict(visible=False, range=[-0.5, 1.8]),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=40, r=80, t=70, b=30),
    )
    return fig


def pdf_safe(text):
    replacements = {
        "–": "-", "—": "-", "→": "->", "←": "<-", "↔": "<->",
        "↓": "v", "↑": "^", "•": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "…": "...", "≥": ">=", "≤": "<=",
        "×": "x", "÷": "/", "°": " degrees",
    }
    if text is None:
        return ""
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def make_pdf(name, date, text, entities, triples, score, conclusion):
    p = FPDF()
    p.set_auto_page_break(True, margin=18)
    p.add_page()
    p.set_font("Helvetica", "B", 16)
    p.cell(0, 10, pdf_safe("Relationship Extraction from Text"), new_x="LMARGIN", new_y="NEXT")
    p.set_font("Helvetica", "", 10)
    p.multi_cell(0, 6, pdf_safe("Outcome: Entity-relationship triples suitable for graph construction."))
    p.ln(4)

    p.set_font("Helvetica", "B", 10)
    p.cell(35, 6, "Student Name:")
    p.set_font("Helvetica", "", 10)
    p.cell(70, 6, pdf_safe(name or "N/A"))
    p.set_font("Helvetica", "B", 10)
    p.ln(7)

    p.set_font("Helvetica", "B", 10)
    p.cell(35, 6, "Date:")
    p.set_font("Helvetica", "", 10)
    p.cell(70, 6, pdf_safe(date))
    p.set_font("Helvetica", "B", 10)
    p.cell(28, 6, "Quiz:")
    p.set_font("Helvetica", "", 10)
    p.cell(50, 6, pdf_safe(f"{score}/{QUIZ_SIZE}"))
    p.ln(12)

    for heading, content in [("1. Aim", AIM), ("2. Input Text", text or "N/A")]:
        p.set_font("Helvetica", "B", 11)
        p.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        p.set_font("Helvetica", "", 9)
        p.multi_cell(0, 5, pdf_safe(content))
        p.ln(3)

    p.set_font("Helvetica", "B", 11)
    p.cell(0, 7, "3. Identified Entities", new_x="LMARGIN", new_y="NEXT")
    p.set_font("Helvetica", "", 9)
    if entities:
        for item in entities:
            p.cell(0, 5, pdf_safe(f"- {item['Entity']} ({item['Type']})"), new_x="LMARGIN", new_y="NEXT")
    else:
        p.cell(0, 5, "No named entities identified.", new_x="LMARGIN", new_y="NEXT")
    p.ln(3)

    p.set_font("Helvetica", "B", 11)
    p.cell(0, 7, "4. Extracted Triples", new_x="LMARGIN", new_y="NEXT")
    if triples:
        widths = [58, 55, 65]
        p.set_font("Helvetica", "B", 8)
        for heading, width in zip(["Subject", "Relation", "Object"], widths):
            p.cell(width, 6, heading, border=1, align="C")
        p.ln()
        p.set_font("Helvetica", "", 8)
        for triple in triples:
            for value, width in zip(
                [triple["Subject"], triple["Relation"], triple["Object"]], widths
            ):
                p.cell(width, 6, pdf_safe(str(value)[:32]), border=1, align="C")
            p.ln()
    else:
        p.set_font("Helvetica", "", 9)
        p.cell(0, 5, "No triples extracted.", new_x="LMARGIN", new_y="NEXT")

    p.ln(5)
    p.set_font("Helvetica", "B", 11)
    p.cell(0, 7, "5. Result / Conclusion", new_x="LMARGIN", new_y="NEXT")
    p.set_font("Helvetica", "", 9)
    p.multi_cell(0, 5, pdf_safe(conclusion))
    return bytes(p.output())


def extract_uploaded_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is required for PDF uploads.")
        reader = PdfReader(BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("python-docx is required for DOCX uploads.")
        doc = Document(BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type. Use TXT, PDF, or DOCX.")


def answer_query(query, triples):
    if not triples:
        return []
    q = query.lower().strip()

    if q.startswith("relationships of:"):
        entity = q.split(":", 1)[1].strip()
        return [t for t in triples if entity in t["Subject"].lower() or entity in t["Object"].lower()]

    if q.startswith("relation between:"):
        pair = q.split(":", 1)[1].strip()
        parts = [x.strip() for x in pair.split("|")]
        if len(parts) != 2:
            return []
        a, b = parts
        return [
            t for t in triples
            if (a in t["Subject"].lower() and b in t["Object"].lower())
            or (b in t["Subject"].lower() and a in t["Object"].lower())
        ]

    if q.startswith("who/what is connected by:"):
        relation_name = q.split(":", 1)[1].strip()
        return [t for t in triples if relation_name in t["Relation"].lower()]

    if q.startswith("where is:"):
        entity = q.split(":", 1)[1].strip()
        return [
            t for t in triples
            if entity in t["Subject"].lower() or entity in t["Object"].lower()
        ]

    # Query 5 / fallback: all triples
    return list(triples)


def query_label_options(triples):
    entities = []
    relations = []
    for t in triples:
        for n in (t["Subject"], t["Object"]):
            if n not in entities:
                entities.append(n)
        if t["Relation"] not in relations:
            relations.append(t["Relation"])

    a = entities[0] if entities else "Google"
    b = entities[1] if len(entities) > 1 else "Gemini"
    r = relations[0] if relations else "developed"
    return [
        f"Relationships of: {a}",
        f"Relation between: {a} | {b}",
        f"Who/What is connected by: {r}",
        f"Where is: {a}",
        "Show all",
    ]


def init():
    defaults = {
        "text": DEFAULT,
        "ents": [],
        "candidate_pairs": [],
        "triples": [],
        "done": False,
        "trials": [],
        "score": 0,
        "quiz": False,
        "quiz_questions": random.sample(QUIZ_BANK, QUIZ_SIZE),
        "method": "spaCy NER + linguistic relation",
        "query": "",
        "query_matches": [],
        "query_step": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def purpose():
    st.title("Purpose")
    st.caption("Virtual Laboratory Experiment")
    st.info(f"**Aim:** {AIM}")
    a, b = st.columns(2)
    with a:
        st.subheader("Learning Objectives")
        objectives = [
            "Identify named entities from unstructured text.",
            "Extract semantic relationships between identified entities.",
            "Represent relationships as Subject–Relation–Object triples.",
            "Visualize triples as a directed semantic graph.",
            "Understand real-world applications of relationship extraction.",
        ]
        for i, item in enumerate(objectives, 1):
            st.write(f"**{i}.** {item}")
    with b:
        st.subheader("Expected Outcome")
        st.success(OUTCOME)
        st.subheader("Experiment Pipeline")
        st.code("Text / Document → NER → Entity Pairs → ML/linguistic Relation → Triples → Graph → Queries")

    st.divider()
    st.subheader("Real-World Applications")
    st.write("Relationship extraction converts unstructured language into structured facts that can support search, knowledge graphs, recommendation, biomedical analysis, compliance and social-media analytics.")
    st.plotly_chart(application_graph(), use_container_width=True, key="home_app_graph")

    st.subheader("Experiment Modules")
    st.write("**Theory** — concepts, ML approach and extraction logic.")
    st.write("**Procedure** — step-by-step experiment.")
    st.write("**Simulation** — text/document upload, extraction, graph and interactive queries.")
    st.write("**Self Evaluation** — assess understanding.")
    st.write("**Result & Report** — view result and download report.")


def theory():
    st.title("Theory")
    st.header("Relationship Extraction")
    st.markdown("Relationship Extraction (RE) is an Information Extraction task in NLP that discovers meaningful semantic relationships between entities in text. For example, **Google developed Gemini** can be represented as **(Google, developed, Gemini)**.")
    st.header("Named Entity Recognition (NER)")
    st.markdown("NER identifies and classifies entities such as people, organizations, locations and products. The application uses spaCy's pretrained statistical NER model as the entity-identification layer.")
    st.header("General ML-Based Relation Extraction")
    st.markdown("The simulation also supports a pretrained transformer relation-extraction model, **REBEL**. REBEL frames relation extraction as text generation and can identify relation types without requiring every relation to be manually hard-coded.")
    st.code("Document → Transformer/NER → Candidate entities → Relation prediction → (Subject, Relation, Object) → Graph")
    st.caption("The ML model is loaded only when selected. spaCy mode remains lightweight for normal demonstrations.")
    st.header("Entity–Relationship Triple")
    st.code("(Subject, Relation, Object)")
    st.markdown("Example: **(Sundar Pichai, is CEO of, Google)**. The subject and object become graph nodes, while the relation becomes a directed edge.")
    st.header("Relation Identification")
    st.dataframe(pd.DataFrame([
        ("is CEO of", "is the CEO of"), ("developed", "developed / created / built"),
        ("founded", "founded / established"), ("headquartered in", "headquartered in / based in"),
        ("studied at", "studied at / graduated from"), ("acquired", "acquired / bought")
    ], columns=["Relation", "Example Pattern"]), hide_index=True, use_container_width=True)
    st.header("Key Terms")
    st.dataframe(pd.DataFrame([
        ("Entity", "Named object identified in text."), ("NER", "Named Entity Recognition."),
        ("Relation", "Semantic connection between entities."), ("Subject", "Source entity."),
        ("Object", "Target entity."), ("Triple", "(Subject, Relation, Object)."),
        ("Graph Node", "Entity represented as a node."), ("Graph Edge", "Relationship represented as a directed edge."),
        ("REBEL", "Pretrained transformer model for relation extraction.")
    ], columns=["Term", "Meaning"]), hide_index=True, use_container_width=True)
    st.divider()
    st.header("Recommended E-Books / Online Textbooks")
    for i, (title, url) in enumerate(BOOK_REFERENCES, 1):
        st.markdown(f"**{i}.** {title}")
        st.caption(url)


def procedure():
    st.title("Procedure")
    st.info("Read all steps before starting the simulation.")
    steps = [
        "Read Aim, Theory and Objectives.",
        "Open Simulation.",
        "Choose spaCy linguistic extraction or the ML model (REBEL).",
        "Enter text or upload a TXT, PDF or DOCX document.",
        "Click Extract Relationships.",
        "Observe identified entities and candidate entity pairs.",
        "Observe Subject–Relation–Object triples.",
        "Inspect the graph through the step-by-step stages.",
        "Select a post-extraction query and start its visual graph traversal.",
        "Click Step 1, Step 2 and Step 3 to see nodes and relationship edges highlighted.",
        "Run up to five graph queries and inspect their answers.",
        "Record the experimental trial.",
        "Complete Self Evaluation.",
        "View Result & Report and download the PDF.",
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"**Step {i}:** {s}")
    st.subheader("Suggested First Input")
    st.code(DEFAULT)


def render_query_visualizer(triples):
    st.header("Observation 5: Interactive Graph Query Visualizer")
    st.write("Select a query, start it, then click the steps. The graph highlights the entity nodes and relationship edge involved in the answer.")

    options = query_label_options(triples)
    selected = st.selectbox("Choose a graph query", options, key="query_select")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start / Reset Query", use_container_width=True, key="query_start"):
            st.session_state.query = selected
            st.session_state.query_matches = answer_query(selected, triples)
            st.session_state.query_step = 1
    with c2:
        if st.button("← Previous Step", use_container_width=True, key="query_prev"):
            st.session_state.query_step = max(1, st.session_state.query_step - 1)
    with c3:
        if st.button("Next Step →", use_container_width=True, key="query_next"):
            st.session_state.query_step = min(3, st.session_state.query_step + 1)

    if not st.session_state.query:
        st.info("Click **Start / Reset Query** to begin the visual query traversal.")
        return

    matches = st.session_state.query_matches
    step = st.session_state.query_step

    if not matches:
        st.warning("No matching triples were found for this query.")
        st.plotly_chart(graph(triples, title="Graph — no matching answer"), use_container_width=True, key="query_no_match_graph")
        return

    if step == 1:
        st.subheader("Step 1 — Find the relevant node(s)")
        highlighted_nodes = set()
        for t in matches:
            highlighted_nodes.update([t["Subject"], t["Object"]])
        st.info("The graph first identifies the entity/entities relevant to the query.")
        title = "Query Step 1 — Relevant entities highlighted"
        highlighted_edges = set()
    elif step == 2:
        st.subheader("Step 2 — Follow the relationship edge")
        highlighted_nodes = set()
        highlighted_edges = set()
        for t in matches:
            highlighted_nodes.update([t["Subject"], t["Object"]])
            highlighted_edges.add((t["Subject"], t["Relation"], t["Object"]))
        st.info("The matching relationship edge is highlighted together with its connected nodes.")
        title = "Query Step 2 — Relationship path highlighted"
    else:
        st.subheader("Step 3 — Display the answer")
        highlighted_nodes = set()
        highlighted_edges = set()
        for t in matches:
            highlighted_nodes.update([t["Subject"], t["Object"]])
            highlighted_edges.add((t["Subject"], t["Relation"], t["Object"]))
        st.success(f"Answer found: {len(matches)} matching relationship(s).")
        title = "Query Step 3 — Answer path highlighted"

    st.plotly_chart(
        graph(triples, highlighted_nodes=highlighted_nodes, highlighted_edges=highlighted_edges, title=title),
        use_container_width=True,
        key=f"query_graph_step_{step}",
    )

    st.markdown("**Current query:** " + st.session_state.query)
    if step == 3:
        answer_df = pd.DataFrame(matches)[["Subject", "Relation", "Object"]]
        st.dataframe(answer_df, hide_index=True, use_container_width=True)

    st.caption("Highlighted nodes = entities involved in the answer. Highlighted directed edge = relationship followed by the query.")


def simulation():
    st.title("Simulation")
    st.write("Perform relationship extraction from typed text or an uploaded document. The output is progressively converted into a semantic graph and can then be queried visually.")

    st.subheader("1. Select Extraction Model")
    method = st.radio(
        "Method",
        ["spaCy NER + linguistic relation", "ML model (REBEL)"],
        horizontal=True,
        help="spaCy mode is lightweight. REBEL is a pretrained transformer model and may require additional download time and RAM/CPU.",
        key="extraction_method",
    )
    st.session_state.method = method

    st.subheader("2. Input Source")
    source = st.radio("Choose input", ["Enter text", "Upload document"], horizontal=True, key="input_source")

    if source == "Enter text":
        choice = st.selectbox("Choose a preset example", ["Custom / Current Text"] + list(EXAMPLES.keys()), key="preset_choice")
        if choice != "Custom / Current Text":
            st.session_state.text = EXAMPLES[choice]
        text = st.text_area("Input Text", value=st.session_state.text, height=160, key="input_text")
    else:
        uploaded = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"], help="Supported formats: TXT, PDF and DOCX.", key="document_uploader")
        text = st.session_state.text
        if uploaded is not None:
            try:
                text = extract_uploaded_text(uploaded)
                st.session_state.text = text
                st.success(f"Loaded {uploaded.name} — {len(text):,} characters.")
                st.text_area("Extracted Document Text", value=text[:12000], height=180, key="document_preview")
                if len(text) > 12000:
                    st.caption("Preview limited to 12,000 characters; extraction uses the full document text.")
            except Exception as exc:
                st.error(f"Could not read the document: {exc}")
                return

    if st.button("Extract Relationships", type="primary", use_container_width=True, key="extract_button"):
        if not text.strip():
            st.warning("Enter text or upload a readable document first.")
            return
        st.session_state.text = text
        st.session_state.query = ""
        st.session_state.query_matches = []
        st.session_state.query_step = 0
        with st.spinner("Running the selected NLP/ML extraction model..."):
            try:
                ents, candidate_pairs, triples = extract(text, method)
                st.session_state.ents = ents
                st.session_state.candidate_pairs = candidate_pairs
                st.session_state.triples = triples
                st.session_state.done = True
            except Exception as exc:
                st.error("Extraction failed. If REBEL is selected, check transformers/torch/sentencepiece and model download access.")
                st.exception(exc)
                return

    if not st.session_state.done:
        st.warning("Enter text or upload a document and click Extract Relationships.")
        return

    st.divider()
    st.header("Observation 1: Identified Entities")
    st.dataframe(pd.DataFrame(st.session_state.ents), hide_index=True, use_container_width=True)

    st.header("Observation 2: Candidate Entity Pairs")
    if st.session_state.candidate_pairs:
        st.dataframe(pd.DataFrame(st.session_state.candidate_pairs), hide_index=True, use_container_width=True)
    else:
        st.info("No entity pairs were generated.")

    st.header("Observation 3: Extracted Entity–Relationship Triples")
    triples = st.session_state.triples
    if triples:
        df = pd.DataFrame(triples)
        st.dataframe(df[["Subject", "Relation", "Object"]], hide_index=True, use_container_width=True)
        st.success(f"{len(df)} relationship triple(s) extracted using {method}.")
    else:
        st.warning("No relation found. Try text containing clear relationships between entities.")

    st.divider()
    st.header("Observation 4: Step-by-Step Graph Simulation")
    if triples:
        nodes = []
        for t in triples:
            for n in (t["Subject"], t["Object"]):
                if n not in nodes:
                    nodes.append(n)

        st.subheader("Step 1 — Entities become graph nodes")
        st.write("Nodes: " + ", ".join(nodes))
        st.plotly_chart(graph(triples[:0], title="Step 1 — Entity nodes identified", standalone_nodes=nodes), use_container_width=True, key="graph_stage_1")

        st.subheader("Step 2 — First relationship becomes a directed edge")
        st.plotly_chart(graph(triples[:1], title="Step 2 — First relationship"), use_container_width=True, key="graph_stage_2")

        st.subheader("Step 3 — Additional triples expand the graph")
        for i in range(1, len(triples)):
            with st.expander(f"Add triple {i + 1}: {triples[i]['Subject']} → {triples[i]['Object']}"):
                st.write(f"Relation: **{triples[i]['Relation']}**")
                st.plotly_chart(graph(triples[: i + 1], title=f"Step 3.{i} — Graph after triple {i + 1}"), use_container_width=True, key=f"graph_stage_3_{i}")

        st.subheader("Step 4 — Final semantic graph")
        st.plotly_chart(graph(triples, title="Step 4 — Final semantic graph"), use_container_width=True, key="graph_stage_final")
        st.caption("Subject/Object = graph nodes; Relation = directed edge.")

        st.divider()
        render_query_visualizer(triples)
    else:
        st.info("The graph and query visualizer appear after at least one relationship triple is extracted.")

    st.divider()
    st.header("Record Experimental Trial")
    if st.button("Record Current Trial", key="record_trial"):
        st.session_state.trials.append({
            "Trial": len(st.session_state.trials) + 1,
            "Model": method,
            "Entities": len(st.session_state.ents),
            "Triples": len(st.session_state.triples),
            "Input": st.session_state.text[:90],
            "Time": datetime.now().strftime("%H:%M:%S"),
        })
        st.success("Trial recorded.")
    if st.session_state.trials:
        st.dataframe(pd.DataFrame(st.session_state.trials), hide_index=True, use_container_width=True)


def quiz():
    st.title("Quiz")
    st.write(
        f"10 questions are randomly selected from the {len(QUIZ_BANK)}-question bank "
        "when a new app session is loaded."
    )

    questions = st.session_state.quiz_questions
    answers = []

    for i, item in enumerate(questions, 1):
        options = item["options"]
        answer = st.radio(
            f"Q{i}. {item['question']}",
            options,
            index=None,
            key=f"quiz_q_{item['id']}",
        )
        answers.append(answer)

    if st.button("Submit Quiz", type="primary", key="submit_quiz"):
        if any(answer is None for answer in answers):
            st.warning("Please answer all 10 questions before submitting.")
            return

        score = sum(
            options.index(answer) == item["answer"]
            for answer, item in zip(answers, questions)
            for options in [item["options"]]
        )

        st.session_state.score = score
        st.session_state.quiz = True
        st.info(f"Final Score: {score}/{QUIZ_SIZE} ({score / QUIZ_SIZE * 100:.0f}%)")

        st.subheader("Answer Review")
        for i, (answer, item) in enumerate(zip(answers, questions), 1):
            correct = item["options"][item["answer"]]
            if answer == correct:
                st.write(f"**Q{i}: Correct**")
            else:
                st.write(f"**Q{i}: Correct answer — {correct}**")



def report_generation():
    st.title("Report Generation")
    if not st.session_state.done:
        st.warning("Perform the Simulation first.")
    else:
        st.success(f"The experiment identified {len(st.session_state.ents)} named entities and extracted {len(st.session_state.triples)} relationship triple(s).")
        if st.session_state.triples:
            st.dataframe(pd.DataFrame(st.session_state.triples)[["Subject", "Relation", "Object"]], hide_index=True, use_container_width=True)
        st.subheader("Conclusion")
        st.write("The experiment successfully extracted semantic relationships from unstructured text and represented them as Subject–Relation–Object triples suitable for graph construction.")

    st.divider()
    st.header("Generate Report")
    name = st.text_input("Student Name")
    date = st.date_input("Experiment Date", value=datetime.now().date())
    conclusion = st.text_area("Observation / Conclusion", value="The experiment successfully extracted semantic relationships from unstructured text. The identified entities and their relationships were represented as Subject–Relation–Object triples and visualized as a directed semantic graph.", height=100)
    pdf = make_pdf(name, str(date), st.session_state.text, st.session_state.ents, st.session_state.triples, st.session_state.score, conclusion)
    st.download_button("Download Experiment Report (PDF)", pdf, "Relationship_Extraction_Virtual_Lab_Report.pdf", "application/pdf", type="primary", use_container_width=True)


def make_certificate_pdf(student_name, certificate_date):
    """Create an individual academic-style Virtual Laboratory certificate.

    This is a locally generated academic project certificate and is not an
    official certificate issued by IIT Kharagpur, the Ministry of Education,
    or any other institution.
    """
    cert = FPDF(orientation="L", unit="mm", format="A4")
    cert.set_auto_page_break(False)
    cert.add_page()

    # Professional certificate border.
    cert.set_draw_color(34, 55, 75)
    cert.set_line_width(1.4)
    cert.rect(8, 8, 281, 194)
    cert.set_line_width(0.5)
    cert.rect(13, 13, 271, 184)

    # Subtle inner accent lines.
    cert.set_draw_color(145, 115, 45)
    cert.set_line_width(0.7)
    cert.line(28, 30, 262, 30)
    cert.line(28, 174, 262, 174)

    # Header.
    cert.set_text_color(32, 49, 66)
    cert.set_font("Helvetica", "B", 12)
    cert.cell(
        0, 8, "VIRTUAL LABORATORY",
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    cert.set_font("Helvetica", "B", 25)
    cert.cell(
        0, 14, "CERTIFICATE OF COMPLETION",
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    cert.set_font("Helvetica", "", 11)
    cert.cell(
        0, 8, "Academic Virtual Laboratory Experiment",
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    cert.ln(10)

    # Main statement.
    cert.set_font("Helvetica", "", 12)
    cert.multi_cell(
        0,
        8,
        pdf_safe(
            "This is to certify that"
        ),
        align="C",
    )

    # Student name is the only required person-specific field.
    cert.set_text_color(26, 45, 62)
    cert.set_font("Helvetica", "B", 24)
    cert.cell(
        0, 14, pdf_safe(student_name or "Student Name"),
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    # Underline for name, similar to formal certificate layouts.
    name_width = min(max(len(student_name or "Student Name") * 4.2 + 16, 55), 180)
    cert.set_draw_color(70, 70, 70)
    cert.set_line_width(0.6)
    cert.line(
        (297 - name_width) / 2,
        88,
        (297 + name_width) / 2,
        88,
    )

    cert.set_text_color(32, 49, 66)
    cert.set_font("Helvetica", "", 12)
    cert.ln(7)
    cert.multi_cell(
        0,
        8,
        pdf_safe(
            "has successfully completed the virtual laboratory experiment on"
        ),
        align="C",
    )

    cert.set_font("Helvetica", "B", 19)
    cert.cell(
        0, 11, pdf_safe("Relationship Extraction from Text"),
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    cert.set_font("Helvetica", "", 10.5)
    cert.ln(4)
    cert.multi_cell(
        0,
        6,
        pdf_safe(
            "The experiment covered semantic relationship extraction, named entity "
            "recognition, Subject-Relation-Object triples, graph visualization, "
            "graph-based querying and self-evaluation."
        ),
        align="C",
    )

    # Completion/date metadata.
    cert.ln(9)
    cert.set_font("Helvetica", "B", 10)
    cert.cell(82, 6, "Date of Completion", align="C")
    cert.cell(82, 6, "Assessment", align="C")
    cert.cell(82, 6, "Experiment Status", align="C")
    cert.ln(6)

    cert.set_font("Helvetica", "", 10)
    cert.cell(82, 6, pdf_safe(str(certificate_date)), align="C")
    cert.cell(82, 6, "Simulation + Quiz", align="C")
    cert.cell(82, 6, "Completed", align="C")

    # Signature lines.
    cert.ln(16)
    left_x = 42
    right_x = 185
    y = 181

    cert.set_draw_color(70, 70, 70)
    cert.set_line_width(0.5)
    cert.line(left_x, y, left_x + 70, y)
    cert.line(right_x, y, right_x + 70, y)

    cert.set_y(y + 2)
    cert.set_x(left_x)
    cert.set_font("Helvetica", "", 9)
    cert.cell(70, 5, "Faculty / Lab Coordinator", align="C")
    cert.set_x(right_x)
    cert.cell(70, 5, "Virtual Laboratory", align="C")

    # Footer note.
    cert.set_y(191)
    cert.set_font("Helvetica", "I", 7)
    cert.cell(
        0,
        4,
        "Generated by the Relationship Extraction Virtual Laboratory | Academic project certificate",
        align="C",
    )

    return bytes(cert.output())


def certificate():
    st.title("Certificate")
    st.write(
        "Generate an individual Certificate of Completion after completing the "
        "virtual laboratory experiment and self-evaluation."
    )

    st.subheader("Experiment")
    st.write("**Relationship Extraction from Text**")

    simulation_done = st.session_state.done
    quiz_done = st.session_state.quiz

    st.subheader("Completion Status")
    st.write("Simulation: " + ("Completed" if simulation_done else "Pending"))
    st.write("Quiz: " + ("Completed" if quiz_done else "Pending"))

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter the name of the student who completed the experiment",
    )

    certificate_date = st.date_input(
        "Certificate Date",
        value=datetime.now().date(),
        key="certificate_date",
    )

    if simulation_done and quiz_done:
        if student_name.strip():
            cert_pdf = make_certificate_pdf(
                student_name.strip(),
                certificate_date,
            )
            st.success(
                "Completion requirements satisfied. Your individual certificate is ready."
            )
            st.download_button(
                "Download Certificate (PDF)",
                cert_pdf,
                "Relationship_Extraction_Virtual_Lab_Certificate.pdf",
                "application/pdf",
                type="primary",
                use_container_width=True,
                key="download_certificate",
            )
        else:
            st.warning("Enter the student's name before generating the certificate.")
    else:
        st.warning(
            "Complete the Simulation and Quiz before generating the completion certificate."
        )

def refs():
    st.title("References")
    st.header("E-Books / Online Textbooks")
    for i, (title, url) in enumerate(BOOK_REFERENCES, 1):
        st.markdown(f"**{i}.** {title}")
        st.caption(url)

    st.header("Research Papers / Academic References")
    for i, (title, url) in enumerate(PAPER_REFERENCES, 1):
        st.markdown(f"**{i}.** {title}")
        st.caption(url)

    st.header("Documentation / Technical Blogs")
    for i, (title, url) in enumerate(BLOG_REFERENCES, 1):
        st.markdown(f"**{i}.** {title}")
        st.caption(url)

    st.divider()
    st.info("Neo4j is not required. The experiment generates structured triples and visualizes a small directed graph directly in the Streamlit application.")


def main():
    init()
    st.sidebar.title("Virtual Lab")
    st.sidebar.markdown("### Relationship Extraction")
    page = st.sidebar.radio(
        "Experiment Navigation",
        ["Purpose", "Theory", "Simulation", "Quiz", "Report Generation", "Certificate", "References"],
    )
    st.sidebar.divider()
    st.sidebar.subheader("Experiment Status")
    st.sidebar.write("Simulation: " + ("Completed" if st.session_state.done else "Pending"))
    st.sidebar.write(f"Trials recorded: {len(st.session_state.trials)}")
    st.sidebar.write("Quiz: " + ("Completed" if st.session_state.quiz else "Pending"))

    pages = {
        "Purpose": purpose,
        "Theory": theory,
        "Simulation": simulation,
        "Quiz": quiz,
        "Report Generation": report_generation,
        "Certificate": certificate,
        "References": refs,
    }
    pages[page]()


if __name__ == "__main__":
    main()
