import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import re

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

QUIZ = [
    (
        "What is the main purpose of relationship extraction?",
        [
            "Identify semantic relationships between entities",
            "Translate text",
            "Remove punctuation",
            "Generate random text",
        ],
        0,
    ),
    (
        "What is the main structured output?",
        ["Subject–Relation–Object triple", "Only nouns", "Audio file", "Image matrix"],
        0,
    ),
    (
        "What does NER identify?",
        ["Named entities and their types", "Only edges", "Quiz answers", "PDF pages"],
        0,
    ),
    (
        "For 'Google developed Gemini', which is correct?",
        [
            "(Google, developed, Gemini)",
            "(Gemini, developed, Google)",
            "(Google, Google, Gemini)",
            "(developed, Google, Gemini)",
        ],
        0,
    ),
    (
        "In the graph, the relationship is represented by a:",
        ["Node", "Directed edge", "Document", "Token"],
        1,
    ),
    (
        "Why are triples suitable for graph construction?",
        [
            "They provide structured entities and relationships",
            "They remove relationships",
            "They contain punctuation",
            "They cannot be visualized",
        ],
        0,
    ),
]

BOOK_REFERENCES = [
    (
        "Jurafsky, D. & Martin, J. H. (2026). Speech and Language Processing, "
        "3rd ed. Online manuscript, Stanford University.",
        "https://web.stanford.edu/~jurafsky/slp3/",
    ),
    (
        "Tunstall, L., von Werra, L. & Wolf, T. (2022). Natural Language Processing "
        "with Transformers, Revised Edition. O'Reilly Media.",
        "https://www.oreilly.com/library/view/natural-language-processing/9781098136789/",
    ),
    (
        "Jain, S. M. (2022). Introduction to Transformers for NLP. Apress.",
        "https://www.oreilly.com/library/view/introduction-to-transformers/9781484288443/",
    ),
]

PAPER_REFERENCES = [
    (
        "Huguet Cabot, P.-L. & Navigli, R. (2021). REBEL: Relation Extraction "
        "By End-to-end Language generation. Findings of EMNLP 2021, 2370–2381.",
        "https://aclanthology.org/2021.findings-emnlp.204/",
    ),
    (
        "Jurafsky, D. & Martin, J. H. (2026). Information Extraction: Relations, "
        "Events, and Time, Chapter 20 of Speech and Language Processing.",
        "https://web.stanford.edu/~jurafsky/slp3/20.pdf",
    ),
]

BLOG_REFERENCES = [
    (
        "spaCy Usage Documentation — spaCy 101 / Named Entity Recognition.",
        "https://spacy.io/usage/spacy-101/",
    ),
    (
        "Streamlit Documentation — st.file_uploader.",
        "https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader",
    ),
]


@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error(
            "spaCy model missing. Run: python -m spacy download en_core_web_sm"
        )
        st.stop()


@st.cache_resource
def load_rebel_model():
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError(
            "Transformers is not installed. Run: pip install transformers torch sentencepiece"
        )

    model_name = "Babelscape/rebel-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
    """General spaCy NER + dependency/lexical relation fallback."""
    doc = nlp(text)
    ents = []
    seen = set()
    triples = []
    tseen = set()
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
                    {
                        "Entity 1": e1.text,
                        "Entity 2": e2.text,
                        "Sentence": sent.text.strip(),
                    }
                )

                r = relation(e1, e2, sent)
                if not r:
                    continue

                sub, obj = (
                    (e1, e2) if e1.start < e2.start else (e2, e1)
                )
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
    """
    Parse REBEL's generated relation format:
    <triplet> subject <subj> relation <obj> object
    """
    text = text.replace("<s>", "").replace("</s>", "").strip()
    pattern = re.compile(
        r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=\s*<triplet>|$)",
        re.IGNORECASE,
    )

    triples = []
    for subject, relation_text, obj in pattern.findall(text):
        subject = subject.strip()
        relation_text = relation_text.strip()
        obj = obj.strip()

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
    """ML-based general relation extraction using the pretrained REBEL model."""
    tokenizer, model = load_rebel_model()

    # Keep chunks moderate for CPU/GPU memory and model context.
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sentences:
        sentences = [text[:4000]]

    all_triples = []
    seen = set()

    for sent in sentences:
        inputs = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=3,
            num_return_sequences=1,
        )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=False,
        )[0]

        parsed = parse_rebel_output(decoded)

        for t in parsed:
            key = (
                t["Subject"].lower(),
                t["Relation"].lower(),
                t["Object"].lower(),
            )
            if key not in seen:
                seen.add(key)
                t["Source Sentence"] = sent
                all_triples.append(t)

    # Use spaCy NER for entity display while relations come from ML model.
    doc = nlp(text)
    ents = []
    seen_entities = set()

    for e in doc.ents:
        key = (e.text.lower(), e.label_)
        if key not in seen_entities:
            seen_entities.add(key)
            ents.append({"Entity": e.text, "Type": e.label_})

    # Add entities occurring in REBEL triples even if spaCy did not label them.
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
                    {
                        "Entity 1": local_ents[i],
                        "Entity 2": local_ents[j],
                        "Sentence": sent,
                    }
                )

    return ents, candidate_pairs, all_triples


def extract(text, method):
    if method == "ML model (REBEL)":
        return extract_rebel(text)
    return extract_spacy(text)


def graph(triples, highlighted=None):
    highlighted = highlighted or []
    nodes = []

    for t in triples:
        for n in (t["Subject"], t["Object"]):
            if n not in nodes:
                nodes.append(n)

    if not nodes:
        return go.Figure()

    pos = {
        n: (i * 2.5, 0 if i % 2 == 0 else 0.7)
        for i, n in enumerate(nodes)
    }

    fig = go.Figure()

    for t in triples:
        x1, y1 = pos[t["Subject"]]
        x2, y2 = pos[t["Object"]]
        is_highlight = (
            t["Subject"], t["Relation"], t["Object"]
        ) in highlighted

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(width=4 if is_highlight else 2),
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.add_annotation(
            x=(x1 + x2) / 2,
            y=(y1 + y2) / 2 + 0.15,
            text=t["Relation"],
            showarrow=False,
        )

        fig.add_annotation(
            x=x2,
            y=y2,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=2,
        )

    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            text=nodes,
            textposition="bottom center",
            marker=dict(size=42, line=dict(width=2)),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Extracted Entity–Relationship Graph",
        height=480,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def application_graph():
    applications = [
        "Knowledge Graphs",
        "Search & QA",
        "Recommendation",
        "Fraud / Compliance",
        "Biomedical NLP",
        "Social Media Analysis",
    ]

    x = [0] + [1] * len(applications)
    y = [0] + list(range(len(applications)))
    labels = ["Relationship\nExtraction"] + applications

    fig = go.Figure()

    for i in range(1, len(applications) + 1):
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, i - 1],
                mode="lines",
                line=dict(width=2),
                showlegend=False,
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=labels,
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
        "–": "-",
        "—": "-",
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "↓": "v",
        "↑": "^",
        "•": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "≥": ">=",
        "≤": "<=",
        "×": "x",
        "÷": "/",
        "°": " degrees",
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


def make_pdf(name, roll, date, text, entities, triples, score, conclusion):
    p = FPDF()
    p.set_auto_page_break(True, margin=18)
    p.add_page()

    p.set_font("Helvetica", "B", 16)
    p.cell(
        0,
        10,
        pdf_safe("Relationship Extraction from Text"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    p.set_font("Helvetica", "", 10)
    p.multi_cell(
        0,
        6,
        pdf_safe("Outcome: Entity-relationship triples suitable for graph construction."),
    )
    p.ln(4)

    p.set_font("Helvetica", "B", 10)
    p.cell(35, 6, "Student Name:")
    p.set_font("Helvetica", "", 10)
    p.cell(70, 6, pdf_safe(name or "N/A"))

    p.set_font("Helvetica", "B", 10)
    p.cell(28, 6, "Roll No.:")
    p.set_font("Helvetica", "", 10)
    p.cell(50, 6, pdf_safe(roll or "N/A"))
    p.ln(7)

    p.set_font("Helvetica", "B", 10)
    p.cell(35, 6, "Date:")
    p.set_font("Helvetica", "", 10)
    p.cell(70, 6, pdf_safe(date))

    p.set_font("Helvetica", "B", 10)
    p.cell(28, 6, "Quiz:")
    p.set_font("Helvetica", "", 10)
    p.cell(50, 6, pdf_safe(f"{score}/{len(QUIZ)}"))
    p.ln(12)

    sections = [
        ("1. Aim", AIM),
        ("2. Input Text", text or "N/A"),
    ]

    for heading, content in sections:
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
            p.cell(
                0,
                5,
                pdf_safe(f"- {item['Entity']} ({item['Type']})"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
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
            values = [
                triple["Subject"],
                triple["Relation"],
                triple["Object"],
            ]
            for value, width in zip(values, widths):
                p.cell(
                    width,
                    6,
                    pdf_safe(str(value)[:32]),
                    border=1,
                    align="C",
                )
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
        return "No extracted triples are available for this query."

    q = query.lower().strip()

    # Query 1: all relationships involving an entity
    if q.startswith("relationships of:"):
        entity = q.split(":", 1)[1].strip()
        matches = [
            t for t in triples
            if entity in t["Subject"].lower()
            or entity in t["Object"].lower()
        ]
        if not matches:
            return f"No relationship found for '{entity}'."
        return "\n".join(
            f"- ({t['Subject']}, {t['Relation']}, {t['Object']})"
            for t in matches
        )

    # Query 2: relation between two entities
    if q.startswith("relation between:"):
        pair = q.split(":", 1)[1].strip()
        parts = [x.strip() for x in pair.split("|")]
        if len(parts) != 2:
            return "Use the format: Entity A | Entity B"

        a, b = parts
        matches = [
            t for t in triples
            if (
                a in t["Subject"].lower()
                and b in t["Object"].lower()
            )
            or (
                b in t["Subject"].lower()
                and a in t["Object"].lower()
            )
        ]
        if not matches:
            return f"No direct relationship found between '{a}' and '{b}'."
        return "\n".join(
            f"- {t['Subject']} --[{t['Relation']}]--> {t['Object']}"
            for t in matches
        )

    # Query 3: objects connected through a relation
    if q.startswith("who/what is connected by:"):
        relation_name = q.split(":", 1)[1].strip()
        matches = [
            t for t in triples
            if relation_name in t["Relation"].lower()
        ]
        if not matches:
            return f"No relationship matching '{relation_name}' was found."
        return "\n".join(
            f"- {t['Subject']} -> {t['Object']}"
            for t in matches
        )

    # Query 4: all triples
    return "\n".join(
        f"- ({t['Subject']}, {t['Relation']}, {t['Object']})"
        for t in triples
    )


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
        "method": "spaCy NER + linguistic relation",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def home():
    st.title(TITLE)
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
        st.code(
            "Text / Document → NER → Entity Pairs → ML/linguistic Relation "
            "Extraction → Triples → Graph → Queries"
        )

    st.divider()
    st.subheader("Real-World Applications")
    st.write(
        "Relationship extraction converts unstructured language into structured "
        "facts that can support search, knowledge graphs, recommendation, "
        "biomedical analysis, compliance and social-media analytics."
    )
    st.plotly_chart(application_graph(), use_container_width=True)

    st.subheader("Experiment Modules")
    st.write("**Theory** — concepts, ML approach and extraction logic.")
    st.write("**Procedure** — step-by-step experiment.")
    st.write("**Simulation** — text/document upload, extraction, graph and queries.")
    st.write("**Self Evaluation** — assess understanding.")
    st.write("**Result & Report** — view result and download report.")


def theory():
    st.title("Theory")
    st.header("Relationship Extraction")
    st.markdown(
        "Relationship Extraction (RE) is an Information Extraction task in NLP "
        "that discovers meaningful semantic relationships between entities in text. "
        "For example, **Google developed Gemini** can be represented as "
        "**(Google, developed, Gemini)**."
    )

    st.header("Named Entity Recognition (NER)")
    st.markdown(
        "NER identifies and classifies entities such as people, organizations, "
        "locations and products. The application uses spaCy's pretrained statistical "
        "NER model as the entity-identification layer."
    )

    st.header("General ML-Based Relation Extraction")
    st.markdown(
        "The simulation now supports a pretrained transformer relation-extraction "
        "model, **REBEL (BART-based)**. REBEL frames relation extraction as text "
        "generation and can extract many relation types without requiring the "
        "application to hard-code every possible relation."
    )
    st.code(
        "Document → Transformer/NER → Candidate entities → Relation prediction "
        "→ (Subject, Relation, Object) → Graph"
    )
    st.caption(
        "The ML model is loaded lazily when selected. This keeps the normal "
        "spaCy mode lightweight while allowing a more general pretrained ML mode."
    )

    st.header("Entity–Relationship Triple")
    st.code("(Subject, Relation, Object)")
    st.markdown(
        "Example: **(Sundar Pichai, is CEO of, Google)**. "
        "The subject and object can become graph nodes, while the relation becomes "
        "a directed edge."
    )

    st.header("Relation Identification")
    st.dataframe(
        pd.DataFrame(
            [
                ("is CEO of", "is the CEO of"),
                ("developed", "developed / created / built"),
                ("founded", "founded / established"),
                ("headquartered in", "headquartered in / based in"),
                ("studied at", "studied at / graduated from"),
                ("acquired", "acquired / bought"),
            ],
            columns=["Relation", "Example Pattern"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.header("Key Terms")
    st.dataframe(
        pd.DataFrame(
            [
                ("Entity", "Named object identified in text."),
                ("NER", "Named Entity Recognition."),
                ("Relation", "Semantic connection between entities."),
                ("Subject", "Source entity."),
                ("Object", "Target entity."),
                ("Triple", "(Subject, Relation, Object)."),
                ("Graph Node", "Entity represented as a node."),
                ("Graph Edge", "Relationship represented as a directed edge."),
                ("REBEL", "Pretrained transformer model for relation extraction."),
            ],
            columns=["Term", "Meaning"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.header("Recommended E-Books / Online Textbooks")
    for title, url in BOOK_REFERENCES:
        st.markdown(f"- {title}  \n  {url}")


def procedure():
    st.title("Procedure")
    st.info("Read all steps before starting the simulation.")

    steps = [
        "Read Aim, Theory and Objectives.",
        "Open Simulation.",
        "Choose the spaCy linguistic method or the ML model (REBEL).",
        "Enter text or upload a TXT, PDF or DOCX document.",
        "Click Extract Relationships.",
        "Observe identified entities and candidate entity pairs.",
        "Observe Subject–Relation–Object triples.",
        "Inspect the graph through the step-by-step stages.",
        "Run the four post-extraction relationship queries.",
        "Record the experimental trial.",
        "Complete Self Evaluation.",
        "View Result & Report and download the PDF.",
    ]

    for i, s in enumerate(steps, 1):
        st.markdown(f"**Step {i}:** {s}")

    st.subheader("Suggested First Input")
    st.code(DEFAULT)


def simulation():
    st.title("Simulation")
    st.write(
        "Perform relationship extraction from typed text or an uploaded document. "
        "The output is progressively converted into a semantic graph."
    )

    st.subheader("1. Select Extraction Model")
    method = st.radio(
        "Method",
        ["spaCy NER + linguistic relation", "ML model (REBEL)"],
        horizontal=True,
        help=(
            "spaCy mode is lightweight and fast. REBEL is a pretrained transformer "
            "model and may require additional download time and RAM/CPU."
        ),
    )
    st.session_state.method = method

    st.subheader("2. Input Source")
    source = st.radio(
        "Choose input",
        ["Enter text", "Upload document"],
        horizontal=True,
    )

    if source == "Enter text":
        choice = st.selectbox(
            "Choose a preset example",
            ["Custom / Current Text"] + list(EXAMPLES.keys()),
        )

        if choice != "Custom / Current Text":
            st.session_state.text = EXAMPLES[choice]

        text = st.text_area(
            "Input Text",
            value=st.session_state.text,
            height=160,
        )
    else:
        uploaded = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF and DOCX.",
        )

        text = st.session_state.text
        if uploaded is not None:
            try:
                text = extract_uploaded_text(uploaded)
                st.session_state.text = text
                st.success(
                    f"Loaded {uploaded.name} — {len(text):,} characters."
                )
                st.text_area(
                    "Extracted Document Text",
                    value=text[:12000],
                    height=180,
                )
                if len(text) > 12000:
                    st.caption(
                        "Preview limited to 12,000 characters; extraction uses the full "
                        "document text."
                    )
            except Exception as exc:
                st.error(f"Could not read the document: {exc}")
                return

    if st.button(
        "Extract Relationships",
        type="primary",
        use_container_width=True,
    ):
        if not text.strip():
            st.warning("Enter text or upload a readable document first.")
            return

        st.session_state.text = text

        with st.spinner(
            "Running the selected NLP/ML extraction model..."
        ):
            try:
                (
                    st.session_state.ents,
                    st.session_state.candidate_pairs,
                    st.session_state.triples,
                ) = extract(text, method)

                st.session_state.done = True

            except Exception as exc:
                st.error(
                    "Extraction failed. If you selected REBEL, make sure the "
                    "transformers/torch dependencies are installed and the model "
                    "can be downloaded."
                )
                st.exception(exc)
                return

    if not st.session_state.done:
        st.warning("Enter text or upload a document and click Extract Relationships.")
        return

    st.divider()
    st.header("Observation 1: Identified Entities")
    st.dataframe(
        pd.DataFrame(st.session_state.ents),
        hide_index=True,
        use_container_width=True,
    )

    st.header("Observation 2: Candidate Entity Pairs")
    if st.session_state.candidate_pairs:
        st.dataframe(
            pd.DataFrame(st.session_state.candidate_pairs),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No entity pairs were generated.")

    st.header("Observation 3: Extracted Entity–Relationship Triples")
    if st.session_state.triples:
        df = pd.DataFrame(st.session_state.triples)
        st.dataframe(
            df[["Subject", "Relation", "Object"]],
            hide_index=True,
            use_container_width=True,
        )
        st.success(
            f"{len(df)} relationship triple(s) extracted using {method}."
        )
    else:
        st.warning(
            "No relation found. Try a text containing clear relationships between entities."
        )

    st.divider()
    st.header("Observation 4: Step-by-Step Graph Simulation")

    triples = st.session_state.triples

    if triples:
        # Stage 1
        st.subheader("Step 1 — Entities become graph nodes")
        nodes = []
        for t in triples:
            for n in (t["Subject"], t["Object"]):
                if n not in nodes:
                    nodes.append(n)
        st.write("Nodes:", ", ".join(nodes))
        st.plotly_chart(
            graph(
                [
                    {
                        "Subject": n,
                        "Relation": "",
                        "Object": n,
                    }
                    for n in []
                ]
            ),
            use_container_width=True,
        ) if False else None

        # Stage 2
        st.subheader("Step 2 — Relations connect pairs")
        first_stage = triples[:1]
        st.plotly_chart(graph(first_stage), use_container_width=True)

        # Stage 3
        st.subheader("Step 3 — Additional triples expand the graph")
        if len(triples) > 1:
            for i in range(1, len(triples)):
                with st.expander(f"Add triple {i + 1}: {triples[i]['Subject']} → {triples[i]['Object']}"):
                    st.write(
                        f"Relation: **{triples[i]['Relation']}**"
                    )
                    st.plotly_chart(
                        graph(triples[: i + 1]),
                        use_container_width=True,
                    )

        # Final graph
        st.subheader("Step 4 — Final semantic graph")
        st.plotly_chart(
            graph(triples),
            use_container_width=True,
        )
        st.caption(
            "Subject/Object = graph nodes; Relation = directed edge."
        )

    st.divider()
    st.header("Observation 5: Four Relationship Queries")
    st.write(
        "Run simple queries against the extracted triples. These queries demonstrate "
        "how structured relationships can be used after graph extraction."
    )

    entity_names = []
    for t in triples:
        for n in (t["Subject"], t["Object"]):
            if n not in entity_names:
                entity_names.append(n)

    relation_names = []
    for t in triples:
        if t["Relation"] not in relation_names:
            relation_names.append(t["Relation"])

    entity_default = entity_names[0] if entity_names else "Google"
    second_entity = entity_names[1] if len(entity_names) > 1 else "Gemini"
    relation_default = relation_names[0] if relation_names else "developed"

    q1 = st.text_input(
        "Query 1 — Relationships of an entity",
        value=f"Relationships of: {entity_default}",
    )
    q2 = st.text_input(
        "Query 2 — Relation between two entities",
        value=f"Relation between: {entity_default} | {second_entity}",
    )
    q3 = st.text_input(
        "Query 3 — Find entities connected by a relation",
        value=f"Who/What is connected by: {relation_default}",
    )
    q4 = st.text_input(
        "Query 4 — Show all extracted triples",
        value="Show all",
    )

    if st.button("Run 4 Queries", use_container_width=True):
        queries = [q1, q2, q3, q4]
        for i, query in enumerate(queries, 1):
            st.subheader(f"Query {i} Result")
            st.info(answer_query(query, triples))

    st.divider()
    st.header("Record Experimental Trial")
    if st.button("Record Current Trial"):
        st.session_state.trials.append(
            {
                "Trial": len(st.session_state.trials) + 1,
                "Model": method,
                "Entities": len(st.session_state.ents),
                "Triples": len(st.session_state.triples),
                "Input": st.session_state.text[:90],
                "Time": datetime.now().strftime("%H:%M:%S"),
            }
        )
        st.success("Trial recorded.")

    if st.session_state.trials:
        st.dataframe(
            pd.DataFrame(st.session_state.trials),
            hide_index=True,
            use_container_width=True,
        )


def quiz():
    st.title("Self Evaluation")
    answers = []

    for i, (q, opts, ans) in enumerate(QUIZ, 1):
        answers.append(
            st.radio(
                f"Q{i}. {q}",
                opts,
                key=f"q{i}",
            )
        )

    if st.button("Submit Quiz", type="primary"):
        score = sum(
            opts.index(a) == ans
            for a, (_, opts, ans) in zip(answers, QUIZ)
        )
        st.session_state.score = score
        st.session_state.quiz = True
        st.info(
            f"Final Score: {score}/{len(QUIZ)} "
            f"({score / len(QUIZ) * 100:.0f}%)"
        )


def result():
    st.title("Result & Report")

    if not st.session_state.done:
        st.warning("Perform the Simulation first.")
    else:
        st.success(
            f"The experiment identified {len(st.session_state.ents)} named entities "
            f"and extracted {len(st.session_state.triples)} relationship triple(s)."
        )

        if st.session_state.triples:
            st.dataframe(
                pd.DataFrame(st.session_state.triples)[
                    ["Subject", "Relation", "Object"]
                ],
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Conclusion")
        st.write(
            "The experiment successfully extracted semantic relationships from "
            "unstructured text and represented them as Subject–Relation–Object "
            "triples suitable for graph construction."
        )

    st.divider()
    st.header("Generate Report")

    a, b = st.columns(2)
    with a:
        name = st.text_input("Student Name")
    with b:
        roll = st.text_input("Roll Number / Student ID")

    date = st.date_input(
        "Experiment Date",
        value=datetime.now().date(),
    )

    conclusion = st.text_area(
        "Observation / Conclusion",
        value=(
            "The experiment successfully extracted semantic relationships from "
            "unstructured text. The identified entities and their relationships "
            "were represented as Subject–Relation–Object triples and visualized "
            "as a directed semantic graph."
        ),
        height=100,
    )

    pdf = make_pdf(
        name,
        roll,
        str(date),
        st.session_state.text,
        st.session_state.ents,
        st.session_state.triples,
        st.session_state.score,
        conclusion,
    )

    st.download_button(
        "Download Experiment Report (PDF)",
        pdf,
        "Relationship_Extraction_Virtual_Lab_Report.pdf",
        "application/pdf",
        type="primary",
        use_container_width=True,
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
    st.info(
        "Neo4j is not required. The experiment generates structured triples and "
        "visualizes a small directed graph directly in the Streamlit application."
    )


def main():
    init()

    st.sidebar.title("Virtual Lab")
    st.sidebar.markdown("### Relationship Extraction")

    page = st.sidebar.radio(
        "Experiment Navigation",
        [
            "Home / Aim",
            "Theory",
            "Procedure",
            "Simulation",
            "Self Evaluation",
            "Result & Report",
            "References",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Experiment Status")
    st.sidebar.write(
        "Simulation: "
        + ("Completed" if st.session_state.done else "Pending")
    )
    st.sidebar.write(f"Trials recorded: {len(st.session_state.trials)}")
    st.sidebar.write(
        "Quiz: "
        + ("Completed" if st.session_state.quiz else "Pending")
    )

    pages = {
        "Home / Aim": home,
        "Theory": theory,
        "Procedure": procedure,
        "Simulation": simulation,
        "Self Evaluation": quiz,
        "Result & Report": result,
        "References": refs,
    }

    pages[page]()


if __name__ == "__main__":
    main()
