import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
from gliner import GLiNER

st.set_page_config(page_title='Relationship Extraction | Virtual Lab', layout='wide')

TITLE='Relationship Extraction from Text'
AIM='To extract semantic relationships between identified entities from unstructured text and represent them as Entity–Relationship triples suitable for graph construction.'
OUTCOME='Entity–relationship triples suitable for graph construction.'
MODEL_NAME='knowledgator/gliner-relex-base-v1.0'

DEFAULT=('Sundar Pichai is the CEO of Google. Google is headquartered in Mountain View. '
         'Google developed Gemini. Sundar Pichai studied at Stanford University.')

EXAMPLES={
    'Google example':DEFAULT,
    'Microsoft–LinkedIn example':'Microsoft acquired LinkedIn. LinkedIn is headquartered in Sunnyvale.',
    'SpaceX example':'Elon Musk founded SpaceX. SpaceX developed Starship.',
    'Complex example':('Marie Curie was born in Warsaw and later moved to Paris. She studied at the University of Paris, '
                       'where she worked with Pierre Curie. The Curies discovered polonium and radium. '
                       'Marie Curie received the Nobel Prize in Physics.'),
    'Technology example':('Apple acquired Beats Electronics in 2014. Tim Cook leads Apple, which is headquartered in Cupertino. '
                          'Apple introduced the iPhone and partnered with OpenAI.')
}

DEFAULT_RELATIONS=[
    'is CEO of','founded','acquired','developed','created','introduced','invented','works for',
    'employed by','located in','headquartered in','born in','died in','studied at','educated at',
    'authored','wrote','directed','acted in','married to','parent of','child of','member of',
    'part of','collaborated with','partnered with','owns','manufactured by','received','won',
    'discovered','associated with'
]

ENTITY_LABELS=['person','organization','company','location','city','country','institution',
               'product','technology','event','date','work','award','other']

QUIZ=[
    ('What is the main purpose of relationship extraction?',['Identify semantic relationships between entities','Translate text','Remove punctuation','Generate random text'],0),
    ('What is the main structured output?',['Subject–Relation–Object triple','Only nouns','Audio file','Image matrix'],0),
    ('What does the transformer model perform jointly?',['Entity and relationship extraction','Only PDF generation','Only graph drawing','Only spelling correction'],0),
    ("For 'Google developed Gemini', which is expected?",['(Google, developed, Gemini)','(Gemini, developed, Google)','(Google, Google, Gemini)','(developed, Google, Gemini)'],0),
    ('In the graph, a relationship is represented by a:',['Node','Directed edge','Document','Token'],1),
    ('Why are triples suitable for graph construction?',['They provide structured entities and relationships','They remove relationships','They contain punctuation','They cannot be visualized'],0)
]

@st.cache_resource(show_spinner=False)
def load_model():
    return GLiNER.from_pretrained(MODEL_NAME)

def relation_labels(text):
    out=[]
    for x in text.replace('\n',',').split(','):
        x=x.strip()
        if x and x.lower() not in {y.lower() for y in out}:
            out.append(x)
    return out

def extract(text, labels):
    model=load_model()
    entities, relations=model.inference(
        texts=[text],
        labels=ENTITY_LABELS,
        relations=labels,
        threshold=0.35,
        adjacency_threshold=0.55,
        relation_threshold=0.55,
        return_relations=True,
        flat_ner=False,
        multi_label=False
    )

    ents=[]
    seen=set()
    for e in entities[0]:
        key=(e['text'].strip().lower(),e['label'].lower())
        if key not in seen:
            seen.add(key)
            ents.append({'Entity':e['text'].strip(),'Type':e['label']})

    triples=[]
    seen=set()
    for r in relations[0]:
        sub=r['head']['text'].strip()
        rel=r['relation'].strip()
        obj=r['tail']['text'].strip()
        key=(sub.lower(),rel.lower(),obj.lower())
        if sub and rel and obj and key not in seen:
            seen.add(key)
            triples.append({
                'Subject':sub,
                'Relation':rel,
                'Object':obj,
                'Source Sentence':text
            })

    return ents,triples

def graph(triples):
    nodes=[]
    for t in triples:
        for n in (t['Subject'],t['Object']):
            if n not in nodes:
                nodes.append(n)

    pos={n:(i*2.5,0 if i%2==0 else .7) for i,n in enumerate(nodes)}
    fig=go.Figure()

    for t in triples:
        x1,y1=pos[t['Subject']]
        x2,y2=pos[t['Object']]

        fig.add_trace(go.Scatter(
            x=[x1,x2],y=[y1,y2],mode='lines',
            line=dict(width=2),showlegend=False,hoverinfo='none'
        ))

        fig.add_annotation(
            x=(x1+x2)/2,y=(y1+y2)/2+.15,
            text=t['Relation'],showarrow=False
        )

        fig.add_annotation(
            x=x2,y=y2,ax=x1,ay=y1,
            xref='x',yref='y',axref='x',ayref='y',
            showarrow=True,arrowhead=3,arrowwidth=2
        )

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode='markers+text',
        text=nodes,textposition='bottom center',
        marker=dict(size=42,line=dict(width=2)),
        showlegend=False
    ))

    fig.update_layout(
        title='Extracted Entity–Relationship Graph',
        height=480,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        margin=dict(l=20,r=20,t=60,b=20)
    )
    return fig

def pdf_safe(text):
    replacements={
        "–":"-","—":"-","→":"->","←":"<-","↔":"<->","↓":"v","↑":"^",
        "•":"-","’":"'","‘":"'","“":'"',"”":'"',"…":"...",
        "≥":">=","≤":"<=","×":"x","÷":"/","°":" degrees"
    }
    if text is None:
        return ""
    text=str(text)
    for old,new in replacements.items():
        text=text.replace(old,new)
    return text.encode("latin-1","replace").decode("latin-1")

class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica','I',8)
        self.cell(0,10,f'Page {self.page_no()}',align='C')

def make_pdf(name,roll,date,text,entities,triples,score,conclusion):
    p=PDF()
    p.alias_nb_pages()
    p.set_auto_page_break(True,margin=18)
    p.add_page()

    p.set_font("Helvetica","B",16)
    p.cell(0,10,pdf_safe("Relationship Extraction from Text"),new_x="LMARGIN",new_y="NEXT")

    p.set_font("Helvetica","",10)
    p.multi_cell(0,6,pdf_safe("Outcome: Entity-relationship triples suitable for graph construction."))
    p.ln(4)

    p.set_font("Helvetica","B",10)
    p.cell(35,6,"Student Name:")
    p.set_font("Helvetica","",10)
    p.cell(70,6,pdf_safe(name or "N/A"))

    p.set_font("Helvetica","B",10)
    p.cell(28,6,"Roll No.:")
    p.set_font("Helvetica","",10)
    p.cell(50,6,pdf_safe(roll or "N/A"))
    p.ln(7)

    p.set_font("Helvetica","B",10)
    p.cell(35,6,"Date:")
    p.set_font("Helvetica","",10)
    p.cell(70,6,pdf_safe(date))

    p.set_font("Helvetica","B",10)
    p.cell(28,6,"Quiz:")
    p.set_font("Helvetica","",10)
    p.cell(50,6,pdf_safe(f"{score}/{len(QUIZ)}"))
    p.ln(12)

    p.set_font("Helvetica","B",11)
    p.cell(0,7,"1. Aim",new_x="LMARGIN",new_y="NEXT")
    p.set_font("Helvetica","",9)
    p.multi_cell(0,5,pdf_safe(AIM))
    p.ln(3)

    p.set_font("Helvetica","B",11)
    p.cell(0,7,"2. Input Text",new_x="LMARGIN",new_y="NEXT")
    p.set_font("Helvetica","",9)
    p.multi_cell(0,5,pdf_safe(text or "N/A"))
    p.ln(3)

    p.set_font("Helvetica","B",11)
    p.cell(0,7,"3. Identified Entities",new_x="LMARGIN",new_y="NEXT")
    p.set_font("Helvetica","",9)

    if entities:
        for item in entities:
            line=f"- {item['Entity']} ({item['Type']})"
            p.cell(0,5,pdf_safe(line),new_x="LMARGIN",new_y="NEXT")
    else:
        p.cell(0,5,"No named entities identified.",new_x="LMARGIN",new_y="NEXT")

    p.ln(3)

    p.set_font("Helvetica","B",11)
    p.cell(0,7,"4. Extracted Triples",new_x="LMARGIN",new_y="NEXT")

    if triples:
        widths=[58,55,65]
        p.set_font("Helvetica","B",8)
        for heading,width in zip(["Subject","Relation","Object"],widths):
            p.cell(width,6,heading,border=1,align="C")
        p.ln()
        p.set_font("Helvetica","",8)

        for triple in triples:
            values=[triple["Subject"],triple["Relation"],triple["Object"]]
            for value,width in zip(values,widths):
                p.cell(width,6,pdf_safe(str(value)[:32]),border=1,align="C")
            p.ln()
    else:
        p.set_font("Helvetica","",9)
        p.cell(0,5,"No triples extracted.",new_x="LMARGIN",new_y="NEXT")

    p.ln(5)
    p.set_font("Helvetica","B",11)
    p.cell(0,7,"5. Result / Conclusion",new_x="LMARGIN",new_y="NEXT")
    p.set_font("Helvetica","",9)
    p.multi_cell(0,5,pdf_safe(conclusion))

    return bytes(p.output())

def init():
    defaults={
        'text':DEFAULT,
        'ents':[],
        'triples':[],
        'done':False,
        'trials':[],
        'score':0,
        'quiz':False,
        'relation_text':', '.join(DEFAULT_RELATIONS)
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v

def home():
    st.title(TITLE)
    st.caption('Virtual Laboratory Experiment')
    st.info(f'**Aim:** {AIM}')

    a,b=st.columns(2)
    with a:
        st.subheader('Learning Objectives')
        for i,x in enumerate([
            'Identify named entities from unstructured text.',
            'Extract semantic relationships between identified entities.',
            'Represent relationships as Subject–Relation–Object triples.',
            'Visualize the triples as a small directed semantic graph.'
        ],1):
            st.write(f'**{i}.** {x}')

    with b:
        st.subheader('Expected Outcome')
        st.success(OUTCOME)
        st.subheader('Experiment Pipeline')
        st.code('Text → NER → Entities → Entity Pairs → Relation → Triple → Graph')

    st.divider()
    st.subheader('Experiment Modules')
    st.write('**Theory** — concepts and extraction logic.')
    st.write('**Procedure** — step-by-step experiment.')
    st.write('**Simulation** — perform relationship extraction.')
    st.write('**Self Evaluation** — assess understanding.')
    st.write('**Result & Report** — view result and download report.')

def theory():
    st.title('Theory')
    st.header('Relationship Extraction')
    st.markdown('Relationship Extraction is an Information Extraction task in NLP that discovers meaningful semantic relationships between entities in text. For example, **Google developed Gemini** gives the triple **(Google, developed, Gemini)**.')

    st.header('Named Entity Recognition (NER)')
    st.markdown('NER identifies and classifies entities before a relationship is extracted. This implementation uses a pretrained transformer-based GLiNER-relex model for joint entity and relationship extraction.')

    st.header('Entity–Relationship Triple')
    st.code('(Subject, Relation, Object)')
    st.markdown('Example: **(Sundar Pichai, is CEO of, Google)**. Subject and object can become graph nodes; the relation becomes a directed edge.')

    st.header('Relation Identification')
    st.dataframe(
        pd.DataFrame([
            ('is CEO of','is the CEO of'),
            ('developed','developed / created / built'),
            ('founded','founded / established'),
            ('headquartered in','headquartered in / based in'),
            ('studied at','studied at / graduated from'),
            ('acquired','acquired / bought')
        ],columns=['Relation','Example Pattern']),
        hide_index=True,use_container_width=True
    )

    st.header('Key Terms')
    st.dataframe(
        pd.DataFrame([
            ('Entity','Named object identified in text.'),
            ('NER','Named Entity Recognition.'),
            ('Relation','Semantic connection between entities.'),
            ('Subject','Source entity.'),
            ('Object','Target entity.'),
            ('Triple','(Subject, Relation, Object).'),
            ('Graph Node','Entity represented as a node.'),
            ('Graph Edge','Relationship represented as a directed edge.')
        ],columns=['Term','Meaning']),
        hide_index=True,use_container_width=True
    )

def procedure():
    st.title('Procedure')
    st.info('Read all steps before starting the simulation.')
    steps=[
        'Read Aim, Theory and Objectives.',
        'Open Simulation.',
        'Enter text containing clear relationships between entities.',
        'Click Extract Relationships.',
        'Observe identified entities and their types.',
        'Observe Subject–Relation–Object triples.',
        'Inspect the semantic graph.',
        'Record the trial and repeat with another input.',
        'Complete Self Evaluation.',
        'View Result & Report and download the PDF.'
    ]
    for i,s in enumerate(steps,1):
        st.markdown(f'**Step {i}:** {s}')
    st.subheader('Suggested First Input')
    st.code(DEFAULT)

def simulation():
    st.title('Simulation')
    st.write('Enter text and perform the relationship extraction experiment.')

    choice=st.selectbox('Choose a preset example',['Custom / Current Text']+list(EXAMPLES.keys()))
    if choice!='Custom / Current Text':
        st.session_state.text=EXAMPLES[choice]

    text=st.text_area('Input Text',value=st.session_state.text,height=150)

    # Kept in the backend only; users see the same simple simulation/output style.
    reltext=st.session_state.relation_text

    if st.button('Extract Relationships',type='primary',use_container_width=True):
        if not text.strip():
            st.warning('Enter text first.')
            return

        labels=relation_labels(reltext)

        with st.spinner('Extracting relationships...'):
            try:
                ents,triples=extract(text,labels)
            except Exception as exc:
                st.error('Relationship extraction failed. Check the model installation.')
                st.code(str(exc))
                return

        st.session_state.text=text
        st.session_state.ents=ents
        st.session_state.triples=triples
        st.session_state.done=True

    if not st.session_state.done:
        st.warning('Enter text and click Extract Relationships.')
        return

    st.divider()
    st.header('Observation 1: Identified Entities')
    if st.session_state.ents:
        st.dataframe(
            pd.DataFrame(st.session_state.ents)[['Entity','Type']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning('No entities identified.')

    st.divider()
    st.header('Observation 2: Entity–Relationship Triples')
    if st.session_state.triples:
        df=pd.DataFrame(st.session_state.triples)
        st.dataframe(
            df[['Subject','Relation','Object']],
            hide_index=True,
            use_container_width=True
        )
        st.success(f"{len(df)} relationship triple(s) extracted.")
    else:
        st.warning('No relation found. Try a clear sentence such as “Google developed Gemini.”')

    st.divider()
    st.header('Observation 3: Semantic Graph')
    if st.session_state.triples:
        st.plotly_chart(graph(st.session_state.triples),use_container_width=True)
        st.caption('Subject/Object = graph nodes; Relation = directed edge.')

    st.divider()
    st.header('Record Experimental Trial')
    if st.button('Record Current Trial'):
        st.session_state.trials.append({
            'Trial':len(st.session_state.trials)+1,
            'Entities':st.session_state.ents,
            'Triples':st.session_state.triples,
            'Input':st.session_state.text[:90],
            'Time':datetime.now().strftime('%H:%M:%S')
        })
        st.success('Trial recorded.')

    if st.session_state.trials:
        st.dataframe(pd.DataFrame(st.session_state.trials),hide_index=True,use_container_width=True)

def quiz():
    st.title('Self Evaluation')
    answers=[]
    for i,(q,opts,ans) in enumerate(QUIZ,1):
        answers.append(st.radio(f'Q{i}. {q}',opts,key=f'q{i}'))

    if st.button('Submit Quiz',type='primary'):
        score=sum(opts.index(a)==ans for a,(_,opts,ans) in zip(answers,QUIZ))
        st.session_state.score=score
        st.session_state.quiz=True
        st.info(f'Final Score: {score}/{len(QUIZ)} ({score/len(QUIZ)*100:.0f}%)')

def result():
    st.title('Result & Report')

    if not st.session_state.done:
        st.warning('Perform the Simulation first.')
    else:
        st.success(f"The experiment identified {len(st.session_state.ents)} named entities and extracted {len(st.session_state.triples)} relationship triple(s).")

        if st.session_state.triples:
            st.dataframe(
                pd.DataFrame(st.session_state.triples)[['Subject','Relation','Object']],
                hide_index=True,
                use_container_width=True
            )

        st.subheader('Conclusion')
        st.write('The experiment successfully extracted semantic relationships from unstructured text and represented them as Subject–Relation–Object triples suitable for graph construction.')

    st.divider()
    st.header('Generate Report')
    a,b=st.columns(2)

    with a:
        name=st.text_input('Student Name')
    with b:
        roll=st.text_input('Roll Number / Student ID')

    date=st.date_input('Experiment Date',value=datetime.now().date())

    conclusion=st.text_area(
        'Observation / Conclusion',
        value='The experiment successfully extracted semantic relationships from unstructured text. The identified entities and their relationships were represented as Subject–Relation–Object triples and visualized as a directed semantic graph.',
        height=100
    )

    pdf=make_pdf(
        name,roll,str(date),st.session_state.text,
        st.session_state.ents,st.session_state.triples,
        st.session_state.score,conclusion
    )

    st.download_button(
        'Download Experiment Report (PDF)',
        pdf,
        'Relationship_Extraction_Virtual_Lab_Report.pdf',
        'application/pdf',
        type='primary',
        use_container_width=True
    )

def refs():
    st.title('References')
    st.markdown(
        '**1.** IIT Kharagpur Virtual Labs — experiment structure and presentation style.\n\n'
        '**2.** GLiNER-relex — pretrained zero-shot joint NER and relation extraction.\n\n'
        '**3.** Streamlit Documentation — application development.\n\n'
        '**4.** Plotly Python Documentation — interactive visualization.'
    )
    st.info('Neo4j is not required. The experiment generates triples and visualizes a small graph directly in the Streamlit frontend, consistent with the given Graph Experiment instruction.')

def main():
    init()
    st.sidebar.title('Virtual Lab')
    st.sidebar.markdown('### Relationship Extraction')

    page=st.sidebar.radio(
        'Experiment Navigation',
        ['Home / Aim','Theory','Procedure','Simulation','Self Evaluation','Result & Report','References']
    )

    st.sidebar.divider()
    st.sidebar.subheader('Experiment Status')
    st.sidebar.write('Simulation: '+('Completed' if st.session_state.done else 'Pending'))
    st.sidebar.write(f"Trials recorded: {len(st.session_state.trials)}")
    st.sidebar.write('Quiz: '+('Completed' if st.session_state.quiz else 'Pending'))

    {
        'Home / Aim':home,
        'Theory':theory,
        'Procedure':procedure,
        'Simulation':simulation,
        'Self Evaluation':quiz,
        'Result & Report':result,
        'References':refs
    }[page]()

if __name__=='__main__':
    main()
