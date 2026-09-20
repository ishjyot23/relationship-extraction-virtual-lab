import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
from gliner import GLiNER
import math

st.set_page_config(page_title='Relationship Extraction | Virtual Lab', layout='wide')

TITLE='Relationship Extraction from Text'
AIM=('To extract semantic relationships between identified entities from unstructured text '
     'and represent them as Entity–Relationship triples suitable for graph construction.')
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
                      'Apple introduced the iPhone and partnered with OpenAI.')}

# These are candidate semantic labels for the zero-shot model, NOT hard-coded sentence rules.
DEFAULT_RELATIONS=[
'is CEO of','founded','acquired','developed','created','introduced','invented','works for',
'employed by','located in','headquartered in','born in','died in','studied at','educated at',
'authored','wrote','directed','acted in','married to','parent of','child of','member of',
'part of','collaborated with','partnered with','owns','manufactured by','received','won',
'discovered','associated with']
ENTITY_LABELS=['person','organization','company','location','city','country','institution',
                'product','technology','event','date','work','award','other']
QUIZ=[
('What is the main purpose of relationship extraction?',['Identify semantic relationships between entities','Translate text','Remove punctuation','Generate random text'],0),
('What is the main structured output?',['Subject–Relation–Object triple','Only nouns','Audio file','Image matrix'],0),
('What does the transformer model perform jointly?',['Entity and relationship extraction','Only PDF generation','Only graph drawing','Only spelling correction'],0),
("For 'Google developed Gemini', which is expected?",['(Google, developed, Gemini)','(Gemini, developed, Google)','(Google, Google, Gemini)','(developed, Google, Gemini)'],0),
('In the graph, a relationship is represented by a:',['Node','Directed edge','Document','Token'],1),
('Why are triples suitable for graph construction?',['They provide structured entities and relationships','They remove relationships','They contain punctuation','They cannot be visualized'],0)]

@st.cache_resource(show_spinner=False)
def load_model():
    return GLiNER.from_pretrained(MODEL_NAME)

def relation_labels(text):
    out=[]
    for x in text.replace('\n',',').split(','):
        x=x.strip()
        if x and x.lower() not in {y.lower() for y in out}: out.append(x)
    return out

def extract(text, labels, entity_threshold=0.35, relation_threshold=0.55):
    model=load_model()
    entities, relations=model.inference(
        texts=[text], labels=ENTITY_LABELS, relations=labels,
        threshold=entity_threshold, adjacency_threshold=0.55,
        relation_threshold=relation_threshold, return_relations=True,
        flat_ner=False, multi_label=False)
    ents=[]; seen=set()
    for e in entities[0]:
        key=(e['text'].strip().lower(),e['label'].lower())
        if key not in seen:
            seen.add(key); ents.append({'Entity':e['text'].strip(),'Type':e['label'],'Confidence':round(float(e.get('score',0)),3)})
    triples=[]; seen=set()
    for r in relations[0]:
        sub=r['head']['text'].strip(); rel=r['relation'].strip(); obj=r['tail']['text'].strip(); score=float(r.get('score',0))
        key=(sub.lower(),rel.lower(),obj.lower())
        if sub and rel and obj and key not in seen:
            seen.add(key); triples.append({'Subject':sub,'Relation':rel,'Object':obj,'Confidence':round(score,3)})
    return ents,triples

def graph(triples):
    nodes=[]
    for t in triples:
        for n in (t['Subject'],t['Object']):
            if n not in nodes: nodes.append(n)
    radius=max(2.8,0.75*len(nodes)); pos={}
    for i,n in enumerate(nodes):
        a=2*math.pi*i/max(len(nodes),1); pos[n]=(radius*math.cos(a),radius*math.sin(a))
    fig=go.Figure()
    for t in triples:
        x1,y1=pos[t['Subject']]; x2,y2=pos[t['Object']]
        fig.add_trace(go.Scatter(x=[x1,x2],y=[y1,y2],mode='lines',line=dict(width=3),showlegend=False,
                                  hovertext=f"{t['Subject']} --[{t['Relation']}]--> {t['Object']}",hoverinfo='text'))
        label=f"<b>{t['Relation']} ({t.get('Confidence',0):.2f})</b>"
        fig.add_annotation(x=(x1+x2)/2,y=(y1+y2)/2,text=label,showarrow=False,
                           font=dict(size=14,color='#111111'),bgcolor='white',bordercolor='#333333',borderwidth=1,borderpad=4)
        fig.add_annotation(x=x2,y=y2,ax=x1,ay=y1,xref='x',yref='y',axref='x',ayref='y',showarrow=True,
                           arrowhead=3,arrowsize=1.3,arrowwidth=3,standoff=24)
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode='markers+text',
                             text=nodes,textposition='middle center',textfont=dict(size=15,color='#111111'),
                             marker=dict(size=62,line=dict(width=3)),showlegend=False,
                             hovertemplate='<b>%{text}</b><extra></extra>'))
    fig.update_layout(title=dict(text='Extracted Entity–Relationship Graph',font=dict(size=22,color='#111111')),
                      height=650,plot_bgcolor='white',paper_bgcolor='white',
                      xaxis=dict(visible=False,range=[-radius-2,radius+2]),yaxis=dict(visible=False,range=[-radius-2,radius+2]),
                      margin=dict(l=30,r=30,t=80,b=30),hoverlabel=dict(bgcolor='white',font_color='#111111'))
    return fig

def pdf_safe(text):
    repl={'–':'-','—':'-','→':'->','←':'<-','↔':'<->','↓':'v','↑':'^','•':'-','’':"'",'‘':"'",'“':'"','”':'"','…':'...','≥':'>=','≤':'<=','×':'x','÷':'/','°':' degrees'}
    text='' if text is None else str(text)
    for a,b in repl.items(): text=text.replace(a,b)
    return text.encode('latin-1','replace').decode('latin-1')

class PDF(FPDF):
    def footer(self):
        self.set_y(-15); self.set_font('Helvetica','I',8); self.cell(0,10,f'Page {self.page_no()}',align='C')

def make_pdf(name,roll,date,text,ents,triples,score,conclusion):
    p=PDF(); p.alias_nb_pages(); p.set_auto_page_break(True,margin=18); p.add_page()
    p.set_font('Helvetica','B',16); p.cell(0,10,pdf_safe(TITLE),new_x='LMARGIN',new_y='NEXT')
    p.set_font('Helvetica','',10); p.multi_cell(0,6,pdf_safe('Outcome: Entity-relationship triples suitable for graph construction.')); p.ln(4)
    p.set_font('Helvetica','B',10); p.cell(35,6,'Student Name:'); p.set_font('Helvetica','',10); p.cell(70,6,pdf_safe(name or 'N/A'))
    p.set_font('Helvetica','B',10); p.cell(25,6,'Roll No.:'); p.set_font('Helvetica','',10); p.cell(50,6,pdf_safe(roll or 'N/A')); p.ln(7)
    p.set_font('Helvetica','B',10); p.cell(35,6,'Date:'); p.set_font('Helvetica','',10); p.cell(70,6,pdf_safe(date))
    p.set_font('Helvetica','B',10); p.cell(25,6,'Quiz:'); p.set_font('Helvetica','',10); p.cell(50,6,f'{score}/{len(QUIZ)}'); p.ln(10)
    for head,body in [('1. Aim',AIM),('2. Input Text',text or 'N/A')]:
        p.set_font('Helvetica','B',11); p.cell(0,7,pdf_safe(head),new_x='LMARGIN',new_y='NEXT'); p.set_font('Helvetica','',9); p.multi_cell(0,5,pdf_safe(body)); p.ln(3)
    p.set_font('Helvetica','B',11); p.cell(0,7,'3. Identified Entities',new_x='LMARGIN',new_y='NEXT'); p.set_font('Helvetica','',9)
    for e in ents: p.cell(0,5,pdf_safe(f"- {e['Entity']} ({e['Type']}, confidence={e.get('Confidence',0):.2f})"),new_x='LMARGIN',new_y='NEXT')
    p.ln(3); p.set_font('Helvetica','B',11); p.cell(0,7,'4. Extracted Entity-Relationship Triples',new_x='LMARGIN',new_y='NEXT')
    widths=[54,58,54,24]; p.set_font('Helvetica','B',8)
    for h,w in zip(['Subject','Relation','Object','Conf.'],widths): p.cell(w,6,h,border=1,align='C')
    p.ln(); p.set_font('Helvetica','',8)
    for t in triples:
        for v,w in zip([t['Subject'],t['Relation'],t['Object'],f"{t.get('Confidence',0):.2f}"],widths): p.cell(w,6,pdf_safe(str(v)[:32]),border=1,align='C')
        p.ln()
    p.ln(5); p.set_font('Helvetica','B',11); p.cell(0,7,'5. Result / Conclusion',new_x='LMARGIN',new_y='NEXT'); p.set_font('Helvetica','',9); p.multi_cell(0,5,pdf_safe(conclusion))
    return bytes(p.output())

def init():
    defaults={'text':DEFAULT,'ents':[],'triples':[],'done':False,'trials':[],'score':0,'quiz':False,'relation_text':', '.join(DEFAULT_RELATIONS)}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def home():
    st.title(TITLE); st.caption('Virtual Laboratory Experiment'); st.info(f'**Aim:** {AIM}')
    a,b=st.columns(2)
    with a:
        st.subheader('Learning Objectives')
        for i,x in enumerate(['Identify entities from unstructured text.','Extract semantic relationships between entities.','Represent relationships as Subject–Relation–Object triples.','Visualize triples as a directed semantic graph.','Understand zero-shot transformer-based relation extraction.'],1): st.write(f'**{i}.** {x}')
    with b:
        st.subheader('Expected Outcome'); st.success(OUTCOME); st.subheader('Experiment Pipeline')
        st.code('Text → Transformer-based Joint NER + Relation Extraction → Entities + Relations → Triples → Directed Graph')
    st.divider(); st.subheader('Technology Stack')
    st.write('**Python | Streamlit | GLiNER-relex transformer | Pandas | Plotly | FPDF | GitHub | Streamlit Community Cloud**')
    st.divider(); st.subheader('Experiment Modules')
    st.write('**Theory** — concepts and transformer extraction logic.'); st.write('**Procedure** — step-by-step experiment.'); st.write('**Simulation** — perform relationship extraction.'); st.write('**Self Evaluation** — assess understanding.'); st.write('**Result & Report** — view result and download report.')

def theory():
    st.title('Theory'); st.header('Relationship Extraction')
    st.markdown('Relationship Extraction identifies semantic connections between entities in natural-language text and converts them into structured facts.')
    st.subheader('Example'); st.code('Input: Google developed Gemini.\n\nTriple: (Google, developed, Gemini)')
    st.header('Transformer-Based Approach')
    st.markdown(f'This Virtual Lab uses **{MODEL_NAME}**, a pretrained transformer-based GLiNER-relex model. It performs joint zero-shot Named Entity Recognition and Relation Extraction. It does not use hard-coded sentence/verb matching rules.')
    st.header('Workflow'); st.code('Unstructured Text → Transformer Encoder → Entities + Relations → (Subject, Relation, Object) → Graph')
    st.header('Relation Labels')
    st.markdown('Relation labels are semantic candidate classes supplied at inference time. A new candidate such as **designed by**, **causes**, or **manufactured by** can be added without writing a new if-statement for each wording.')
    st.header('Important Limitation'); st.warning('A pretrained model is more general than fixed rules, but it is not guaranteed to correctly understand every possible sentence. Predictions depend on the model, labels, thresholds and context.')

def procedure():
    st.title('Procedure'); st.info('Read all steps before starting the simulation.')
    steps=['Read Aim, Theory and Objectives.','Open Simulation.','Select a preset or enter custom text.','Review or add semantic relation labels.','Set entity and relation confidence thresholds.','Click Extract Relationships.','Observe identified entities and confidence scores.','Observe Subject–Relation–Object triples.','Inspect the directed semantic graph.','Record the trial and repeat with a different/complex input.','Complete Self Evaluation.','Open Result & Report and download the PDF.']
    for i,s in enumerate(steps,1): st.markdown(f'**Step {i}:** {s}')
    st.subheader('Suggested Complex Input'); st.code(EXAMPLES['Complex example'])

def simulation():
    st.title('Simulation'); st.write('Enter natural-language text and extract relationships using the pretrained transformer model.')
    choice=st.selectbox('Choose a preset example',['Custom / Current Text']+list(EXAMPLES.keys()))
    if choice!='Custom / Current Text': st.session_state.text=EXAMPLES[choice]
    text=st.text_area('Input Text',value=st.session_state.text,height=180)
    reltext=st.text_area('Semantic Relation Labels (comma-separated)',value=st.session_state.relation_text,height=100,
                          help='These are candidate semantic relation classes, not hard-coded sentence patterns.')
    c1,c2=st.columns(2)
    with c1: et=st.slider('Entity confidence threshold',0.20,0.70,0.35,0.05)
    with c2: rt=st.slider('Relation confidence threshold',0.30,0.90,0.55,0.05)
    st.caption('Lower thresholds can increase recall; higher thresholds generally produce fewer, more confident predictions.')
    if st.button('Extract Relationships',type='primary',use_container_width=True):
        if not text.strip(): st.warning('Enter text first.'); return
        labels=relation_labels(reltext)
        if not labels: st.warning('Enter at least one relation label.'); return
        if len(labels)>60: st.warning('Keep the relation-label list to 60 or fewer for a faster experiment.'); return
        with st.spinner('Running the pretrained transformer model...'):
            try: ents,triples=extract(text,labels,et,rt)
            except Exception as exc:
                st.error('Transformer extraction failed. Check the app logs/model installation.'); st.code(str(exc)); return
        st.session_state.text=text; st.session_state.relation_text=reltext; st.session_state.ents=ents; st.session_state.triples=triples; st.session_state.done=True
    if not st.session_state.done: st.warning('Enter text and click Extract Relationships.'); return
    st.divider(); st.header('Observation 1: Identified Entities')
    if st.session_state.ents: st.dataframe(pd.DataFrame(st.session_state.ents),hide_index=True,use_container_width=True)
    else: st.warning('No entities identified.')
    st.divider(); st.header('Observation 2: Entity–Relationship Triples')
    if st.session_state.triples:
        df=pd.DataFrame(st.session_state.triples); st.dataframe(df[['Subject','Relation','Object','Confidence']],hide_index=True,use_container_width=True)
        st.success(f"{len(df)} relationship triple(s) extracted.")
        st.download_button('Download Extracted Triples (CSV)',df.to_csv(index=False).encode('utf-8'),'relationship_extraction_triples.csv','text/csv',use_container_width=True)
    else: st.warning('No relationship passed the current threshold. Try another label or slightly lower the relation threshold.')
    st.divider(); st.header('Observation 3: Semantic Graph')
    if st.session_state.triples:
        st.plotly_chart(graph(st.session_state.triples),use_container_width=True)
        st.info('Entities = graph nodes; semantic relations = directed edges. Labels are intentionally dark with white backgrounds for readability.')
    st.divider(); st.header('4. Interpretation')
    st.success(f"Extracted {len(st.session_state.triples)} relationship triple(s) from {len(st.session_state.ents)} identified entities.")
    st.write('Each output has the form **(Subject, Relation, Object)** and can directly serve as a structured basis for graph construction.')
    st.divider(); st.header('5. Trial Logger')
    if st.button('Record Current Trial',type='primary'):
        st.session_state.trials.append({'Trial':len(st.session_state.trials)+1,'Entities':len(st.session_state.ents),'Triples':len(st.session_state.triples),'Input Preview':st.session_state.text[:100],'Time':datetime.now().strftime('%H:%M:%S')}); st.success('Current trial recorded.')
    if st.session_state.trials: st.dataframe(pd.DataFrame(st.session_state.trials),hide_index=True,use_container_width=True)

def quiz():
    st.title('Self Evaluation'); answers=[]
    for i,(q,opts,ans) in enumerate(QUIZ,1): answers.append(st.radio(f'Q{i}. {q}',opts,key=f'quiz_{i}'))
    if st.button('Submit Quiz',type='primary'):
        score=sum(opts.index(a)==ans for a,(_,opts,ans) in zip(answers,QUIZ)); st.session_state.score=score; st.session_state.quiz=True; st.info(f'Final Score: {score}/{len(QUIZ)} ({score/len(QUIZ)*100:.0f}%)')

def result():
    st.title('Result & Report')
    if not st.session_state.done: st.warning('Perform the Simulation first.')
    else:
        st.success(f"The experiment identified {len(st.session_state.ents)} entities and extracted {len(st.session_state.triples)} relationship triple(s).")
        if st.session_state.triples: st.dataframe(pd.DataFrame(st.session_state.triples)[['Subject','Relation','Object','Confidence']],hide_index=True,use_container_width=True)
        st.subheader('Conclusion'); st.write('The experiment used a pretrained transformer-based zero-shot model to identify entities and semantic relationships. The results were represented as Subject–Relation–Object triples and visualized as a directed semantic graph, producing structured data suitable for graph construction.')
    st.divider(); st.header('Generate Report'); a,b=st.columns(2)
    with a: name=st.text_input('Student Name')
    with b: roll=st.text_input('Roll Number / Student ID')
    date=st.date_input('Experiment Date',value=datetime.now().date())
    conclusion=st.text_area('Observation / Conclusion',value='The experiment successfully extracted semantic relationships from unstructured text using a pretrained transformer-based zero-shot relation extraction model. The identified entities and their relationships were represented as Subject–Relation–Object triples and visualized as a directed semantic graph.',height=120)
    pdf=make_pdf(name,roll,str(date),st.session_state.text,st.session_state.ents,st.session_state.triples,st.session_state.score,conclusion)
    st.download_button('Download Experiment Report (PDF)',pdf,'Relationship_Extraction_Virtual_Lab_Report.pdf','application/pdf',type='primary',use_container_width=True)

def refs():
    st.title('References')
    st.markdown(f'''**1.** IIT Kharagpur Virtual Labs — experiment structure and presentation style.\n\n**2.** GLiNER-relex — `{MODEL_NAME}`, pretrained zero-shot joint NER and relation extraction.\n\n**3.** Streamlit Documentation — application development and deployment.\n\n**4.** Plotly Python — interactive graph visualization.\n\n**5.** FPDF — PDF report generation.''')
    st.info('Neo4j is not required. The required outcome is the generation of entity–relationship triples suitable for graph construction, so the triples are visualized directly in the Streamlit frontend.')

def main():
    init(); st.sidebar.title('Virtual Lab'); st.sidebar.markdown('### Relationship Extraction')
    page=st.sidebar.radio('Experiment Navigation',['Home / Aim','Theory','Procedure','Simulation','Self Evaluation','Result & Report','References'])
    st.sidebar.divider(); st.sidebar.subheader('Experiment Status'); st.sidebar.write('Simulation: '+('Completed' if st.session_state.done else 'Pending')); st.sidebar.write(f"Trials recorded: {len(st.session_state.trials)}"); st.sidebar.write('Quiz: '+('Completed' if st.session_state.quiz else 'Pending')); st.sidebar.divider(); st.sidebar.caption('Model: GLiNER-relex-base-v1.0')
    {'Home / Aim':home,'Theory':theory,'Procedure':procedure,'Simulation':simulation,'Self Evaluation':quiz,'Result & Report':result,'References':refs}[page]()

if __name__=='__main__': main()
