import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
st.set_page_config(page_title='Relationship Extraction | Virtual Lab', layout='wide')
TITLE='Relationship Extraction from Text'
AIM='To extract semantic relationships between identified entities from unstructured text and represent them as Entity–Relationship triples suitable for graph construction.'
OUTCOME='Entity–relationship triples suitable for graph construction.'
DEFAULT=('Sundar Pichai is the CEO of Google. Google is headquartered in Mountain View. Google developed Gemini. Sundar Pichai studied at Stanford University.')
EXAMPLES={
'Google example',
'Microsoft–LinkedIn example':'Microsoft acquired LinkedIn. LinkedIn is headquartered in Sunnyvale.',
'SpaceX example':'Elon Musk founded SpaceX. SpaceX developed Starship.'}
PATTERNS={
'is CEO of':['is the ceo of','is ceo of','chief executive officer of','ceo of'],
'developed':['developed','created','built','introduced'],
'founded':['founded','co-founded','cofounded','established','started'],
'headquartered in':['headquartered in','based in','located in'],
'studied at':['studied at','graduated from','attended'],
'acquired':['acquired','bought','purchased'],
'works at':['works at','works for','employee of','joined'],
'partnered with':['partnered with','collaborated with','worked with']}
QUIZ=[
('What is the main purpose of relationship extraction?',['Identify semantic relationships between entities','Translate text','Remove punctuation','Generate random text'],0),
('What is the main structured output?',['Subject–Relation–Object triple','Only nouns','Audio file','Image matrix'],0),
('What does NER identify?',['Named entities and their types','Only edges','Quiz answers','PDF pages'],0),
("For 'Google developed Gemini', which is correct?",['(Google, developed, Gemini)','(Gemini, developed, Google)','(Google, Google, Gemini)','(developed, Google, Gemini)'],0),
('In the graph, the relationship is represented by a:',['Node','Directed edge','Document','Token'],1),
('Why are triples suitable for graph construction?',['They provide structured entities and relationships','They remove relationships','They contain punctuation','They cannot be visualized'],0)]
@st.cache_resource
def model():
try spacy.load('en_core_web_sm')
except OSError:
st.error("spaCy model missing. Run: python -m spacy download en_core_web_sm")
st.stop()
nlp=model()
def norm(x) ' '.join(x.lower().split())
def relation(e1,e2,s):
low=norm(s.text); a=min(e1.end_char,e2.end_char); b=max(e1.start_char,e2.start_char); between=norm(s.text[a])
hits=[]
for r,phrases in PATTERNS.items():
for p in phrases:
if p in between or p in low.append((len(p),r))
if hits sorted(hits,reverse=True)[0][1]
lo,hi=min(e1.start,e2.start),max(e1.end,e2.end)
verbs=[t.lemma_ for t in s if lo<=t.i<=hi and t.pos_=='VERB']
return verbs[0] if verbs else None
def extract(text):
doc=nlp(text); ents=[]; seen=set(); triples=[]; tseen=set()
for e in doc.ents:
k=(e.text.lower(),e.label_)
if k not in seen.add(k);ents.append({'Entity'.text,'Type'.label_})
for s in doc.sents:
es=list(s.ents)
for i in range(len(es)):
for j in range(i+1,len(es)):
e1,e2=es[i],es[j]; r=relation(e1,e2,s)
if not r
sub,obj=(e1,e2) if e1.start<e2.start else (e2,e1)
k=(sub.text.lower(),r.lower(),obj.text.lower())
if k not in tseen:
tseen.add(k);triples.append({'Subject'.text,'Relation','Object'.text,'Source Sentence'.text.strip()})
return ents,triples
def graph(triples):
nodes=[]
for t in triples:
for n in (t['Subject'],t['Object']):
if n not in nodes.append(n)
pos={n:(i*2.5,0 if i%2==0 else .7) for i,n in enumerate(nodes)};fig=go.Figure()
for t in triples:
x1,y1=pos[t['Subject']];x2,y2=pos[t['Object']]
fig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines',line=dict(width=2),showlegend=False,hoverinfo='none'))
fig.add_annotation(x=(x1+x2)/2,y=(y1+y2)/2+.15,text=t['Relation'],showarrow=False)
fig.add_annotation(x=x2,y=y2,ax=x1,ay=y1,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=3,arrowwidth=2)
fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode='markers+text',text=nodes,textposition='bottom center',marker=dict(size=42,line=dict(width=2)),showlegend=False))
fig.update_layout(title='Extracted Entity–Relationship Graph',height=480,xaxis=dict(visible=False),yaxis=dict(visible=False),plot_bgcolor='white',margin=dict(l=20,r=20,t=60,b=20));return fig
class PDF(FPDF):
def footer(self):
self.set_y(-15);self.set_font('Helvetica','I',8);self.cell(0,10,f'Page {self.page_no()}',align='C')
def pdf_safe(text):
"""Convert Unicode text to characters supported by FPDF core fonts."""
if text is None:
return ""

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

text = str(text)

for old, new in replacements.items():
    text = text.replace(old, new)

return text.encode("latin-1", "replace").decode("latin-1")

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
    new_y="NEXT"
)

p.set_font("Helvetica", "", 10)

p.multi_cell(
    0,
    6,
    pdf_safe(
        "Outcome: Entity-relationship triples suitable for graph construction."
    )
)

p.ln(4)

# Student information
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

# Aim
p.set_font("Helvetica", "B", 11)
p.cell(
    0,
    7,
    "1. Aim",
    new_x="LMARGIN",
    new_y="NEXT"
)

p.set_font("Helvetica", "", 9)
p.multi_cell(0, 5, pdf_safe(AIM))
p.ln(3)

# Input
p.set_font("Helvetica", "B", 11)
p.cell(
    0,
    7,
    "2. Input Text",
    new_x="LMARGIN",
    new_y="NEXT"
)

p.set_font("Helvetica", "", 9)
p.multi_cell(
    0,
    5,
    pdf_safe(text or "N/A")
)
p.ln(3)

# Entities
p.set_font("Helvetica", "B", 11)
p.cell(
    0,
    7,
    "3. Identified Entities",
    new_x="LMARGIN",
    new_y="NEXT"
)

p.set_font("Helvetica", "", 9)

if entities:
    for item in entities:
        line = f"- {item['Entity']} ({item['Type']})"

        p.cell(
            0,
            5,
            pdf_safe(line),
            new_x="LMARGIN",
            new_y="NEXT"
        )
else:
    p.cell(
        0,
        5,
        "No named entities identified.",
        new_x="LMARGIN",
        new_y="NEXT"
    )

p.ln(3)

# Triples
p.set_font("Helvetica", "B", 11)
p.cell(
    0,
    7,
    "4. Extracted Triples",
    new_x="LMARGIN",
    new_y="NEXT"
)

if triples:

    widths = [58, 55, 65]

    p.set_font("Helvetica", "B", 8)

    for heading, width in zip(
        ["Subject", "Relation", "Object"],
        widths
    ):
        p.cell(
            width,
            6,
            heading,
            border=1,
            align="C"
        )

    p.ln()

    p.set_font("Helvetica", "", 8)

    for triple in triples:

        values = [
            triple["Subject"],
            triple["Relation"],
            triple["Object"]
        ]

        for value, width in zip(values, widths):
            p.cell(
                width,
                6,
                pdf_safe(str(value)[:32]),
                border=1,
                align="C"
            )

        p.ln()

else:

    p.set_font("Helvetica", "", 9)

    p.cell(
        0,
        5,
        "No triples extracted.",
        new_x="LMARGIN",
        new_y="NEXT"
    )

p.ln(5)

# Conclusion
p.set_font("Helvetica", "B", 11)

p.cell(
    0,
    7,
    "5. Result / Conclusion",
    new_x="LMARGIN",
    new_y="NEXT"
)

p.set_font("Helvetica", "", 9)

p.multi_cell(
    0,
    5,
    pdf_safe(conclusion)
)

return bytes(p.output())

def init():
for k,v in {'text','ents':[],'triples':[],'done','trials':[],'score':0,'quiz'}.items():
if k not in st.session_state.session_state[k]=v
def home():
st.title(TITLE);st.caption('Virtual Laboratory Experiment');st.info(f'Aim: {AIM}')
a,b=st.columns(2)
with a:
st.subheader('Learning Objectives')
for i,x in enumerate(['Identify named entities from unstructured text.','Extract semantic relationships between identified entities.','Represent relationships as Subject–Relation–Object triples.','Visualize the triples as a small directed semantic graph.'],1).write(f'{i}. {x}')
with b:
st.subheader('Expected Outcome');st.success(OUTCOME);st.subheader('Experiment Pipeline');st.code('Text → NER → Entities → Entity Pairs → Relation → Triple → Graph')
st.divider();st.subheader('Experiment Modules');st.write('Theory — concepts and extraction logic.');st.write('Procedure — step-by-step experiment.');st.write('Simulation — perform relationship extraction.');st.write('Self Evaluation — assess understanding.');st.write('Result & Report — view result and download report.')
def theory():
st.title('Theory');st.header('Relationship Extraction')
st.markdown('Relationship Extraction is an Information Extraction task in NLP that discovers meaningful semantic relationships between entities in text. For example, Google developed Gemini gives the triple (Google, developed, Gemini).')
st.header('Named Entity Recognition (NER)');st.markdown('NER identifies and classifies entities before a relationship is extracted. This implementation uses spaCy’s pretrained English pipeline.')
st.header('Entity–Relationship Triple');st.code('(Subject, Relation, Object)');st.markdown('Example: (Sundar Pichai, is CEO of, Google). Subject and object can become graph nodes; the relation becomes a directed edge.')
st.header('Relation Identification');st.dataframe(pd.DataFrame([('is CEO of','is the CEO of'),('developed','developed / created / built'),('founded','founded / established'),('headquartered in','headquartered in / based in'),('studied at','studied at / graduated from'),('acquired','acquired / bought')],columns=['Relation','Example Pattern']),hide_index=True,use_container_width=True)
st.header('Key Terms');st.dataframe(pd.DataFrame([('Entity','Named object identified in text.'),('NER','Named Entity Recognition.'),('Relation','Semantic connection between entities.'),('Subject','Source entity.'),('Object','Target entity.'),('Triple','(Subject, Relation, Object).'),('Graph Node','Entity represented as a node.'),('Graph Edge','Relationship represented as a directed edge.')],columns=['Term','Meaning']),hide_index=True,use_container_width=True)
def procedure():
st.title('Procedure');st.info('Read all steps before starting the simulation.')
steps=['Read Aim, Theory and Objectives.','Open Simulation.','Enter text containing clear relationships between entities.','Click Extract Relationships.','Observe identified entities and their types.','Observe Subject–Relation–Object triples.','Inspect the semantic graph.','Record the trial and repeat with another input.','Complete Self Evaluation.','View Result & Report and download the PDF.']
for i,s in enumerate(steps,1).markdown(f'Step {i}: {s}')
st.subheader('Suggested First Input');st.code(DEFAULT)
def simulation():
st.title('Simulation');st.write('Enter text and perform the relationship extraction experiment.')
choice=st.selectbox('Choose a preset example',['Custom / Current Text']+list(EXAMPLES.keys()))
if choice!='Custom / Current Text'.session_state.text=EXAMPLES[choice]
text=st.text_area('Input Text',value=st.session_state.text,height=150)
if st.button('Extract Relationships',type='primary',use_container_width=True):
if not text.strip().warning('Enter text first.');return
st.session_state.text=text;st.session_state.ents,st.session_state.triples=extract(text);st.session_state.done=True
if not st.session_state.done.warning('Enter text and click Extract Relationships.');return
st.divider();st.header('Observation 1: Identified Entities');st.dataframe(pd.DataFrame(st.session_state.ents),hide_index=True,use_container_width=True)
st.divider();st.header('Observation 2: Entity–Relationship Triples')
if st.session_state.triples:
df=pd.DataFrame(st.session_state.triples);st.dataframe(df[['Subject','Relation','Object']],hide_index=True,use_container_width=True);st.success(f"{len(df)} relationship triple(s) extracted.")
else.warning('No relation found. Try a clear sentence such as “Google developed Gemini.”')
st.divider();st.header('Observation 3: Semantic Graph')
if st.session_state.triples.plotly_chart(graph(st.session_state.triples),use_container_width=True);st.caption('Subject/Object = graph nodes; Relation = directed edge.')
st.divider();st.header('Record Experimental Trial')
if st.button('Record Current Trial'):
st.session_state.trials.append({'Trial'(st.session_state.trials)+1,'Entities'(st.session_state.ents),'Triples'(st.session_state.triples),'Input'.session_state.text[:90],'Time'.now().strftime('%H:%M:%S')});st.success('Trial recorded.')
if st.session_state.trials.dataframe(pd.DataFrame(st.session_state.trials),hide_index=True,use_container_width=True)
def quiz():
st.title('Self Evaluation');answers=[]
for i,(q,opts,ans) in enumerate(QUIZ,1).append(st.radio(f'Q{i}. {q}',opts,key=f'q{i}'))
if st.button('Submit Quiz',type='primary'):
score=sum(opts.index(a)==ans for a,(_,opts,ans) in zip(answers,QUIZ));st.session_state.score=score;st.session_state.quiz=True;st.info(f'Final Score: {score}/{len(QUIZ)} ({score/len(QUIZ)*100:.0f}%)')
def result():
st.title('Result & Report')
if not st.session_state.done.warning('Perform the Simulation first.')
else:
st.success(f"The experiment identified {len(st.session_state.ents)} named entities and extracted {len(st.session_state.triples)} relationship triple(s).")
if st.session_state.triples.dataframe(pd.DataFrame(st.session_state.triples)[['Subject','Relation','Object']],hide_index=True,use_container_width=True)
st.subheader('Conclusion');st.write('The experiment successfully extracted semantic relationships from unstructured text and represented them as Subject–Relation–Object triples suitable for graph construction.')
st.divider();st.header('Generate Report');a,b=st.columns(2)
with a=st.text_input('Student Name')
with b=st.text_input('Roll Number / Student ID')
date=st.date_input('Experiment Date',value=datetime.now().date());conclusion=st.text_area('Observation / Conclusion',value='The experiment successfully extracted semantic relationships from unstructured text. The identified entities and their relationships were represented as Subject–Relation–Object triples and visualized as a directed semantic graph.',height=100)
pdf=make_pdf(name,roll,str(date),st.session_state.text,st.session_state.ents,st.session_state.triples,st.session_state.score,conclusion)
st.download_button('Download Experiment Report (PDF)',pdf,'Relationship_Extraction_Virtual_Lab_Report.pdf','application/pdf',type='primary',use_container_width=True)
def refs():
st.title('References');st.markdown('1. IIT Kharagpur Virtual Labs — experiment structure and presentation style.\n\n2. spaCy Documentation — NLP and Named Entity Recognition.\n\n3. Streamlit Documentation — application development.\n\n4. Plotly Python Documentation — interactive visualization.')
st.info('Neo4j is not required. The experiment generates triples and visualizes a small graph directly in the Streamlit frontend, consistent with the given Graph Experiment instruction.')
def main():
init();st.sidebar.title('Virtual Lab');st.sidebar.markdown('### Relationship Extraction')
page=st.sidebar.radio('Experiment Navigation',['Home / Aim','Theory','Procedure','Simulation','Self Evaluation','Result & Report','References'])
st.sidebar.divider();st.sidebar.subheader('Experiment Status');st.sidebar.write('Simulation: '+('Completed' if st.session_state.done else 'Pending'));st.sidebar.write(f"Trials recorded: {len(st.session_state.trials)}");st.sidebar.write('Quiz: '+('Completed' if st.session_state.quiz else 'Pending'))
{'Home / Aim','Theory','Procedure','Simulation','Self Evaluation','Result & Report','References'}page
if name=='main'()
