# dashboard_banco_staff.py
# Dashboard bancário profissional para uso interno (colaboradores)
# - Rede Bayesiana (Risco Operacional + Risco de Crédito)
# - Base sintética de 5.000 clientes com variáveis identificadoras, comerciais e de risco
# - Autenticação simples com papéis (analista / gestor)
# - Abas: Visão Geral (KPIs), Análise de Clientes (filtros + tabelas), Rede Bayesiana (simulação), Alertas & Auditoria
# Requisitos: streamlit, pandas, numpy, pgmpy, plotly, networkx
# Execução: pip install streamlit pandas numpy pgmpy plotly networkx && streamlit run dashboard_banco_staff.py

import numpy as np
if not hasattr(np, "product"):
    np.product = np.prod


import streamlit as st
import pandas as pd
import numpy as np
import random
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
from datetime import datetime

st.set_page_config(page_title='Portal Interno — Dashboard de Risco', layout='wide')

# -----------------------------
# 0. Autenticação simples (demo)
# -----------------------------
USERS = {
    'analyst': {'password': 'analyst123', 'role': 'analyst'},
    'manager': {'password': 'manager123', 'role': 'manager'},
    'auditor': {'password': 'auditor123', 'role': 'auditor'}
}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user'] = None
    st.session_state['role'] = None
    st.session_state['audit_log'] = []

def login():
    username = st.session_state['username_input']
    password = st.session_state['password_input']
    user = USERS.get(username)
    if user and user['password'] == password:
        st.session_state['logged_in'] = True
        st.session_state['user'] = username
        st.session_state['role'] = user['role']
        st.session_state['audit_log'].append((datetime.utcnow().isoformat(), username, 'login'))
    else:
        st.session_state['login_error'] = 'Credenciais inválidas'

with st.sidebar:
    if not st.session_state['logged_in']:
        st.text('Entrar (demo)')
        st.text_input('Usuário', key='username_input')
        st.text_input('Senha', type='password', key='password_input')
        st.button('Entrar', on_click=login)
        if 'login_error' in st.session_state:
            st.error(st.session_state['login_error'])
        st.stop()
    else:
        st.write(f"Utilizador: **{st.session_state['user']}** — Papel: **{st.session_state['role']}**")
        if st.button('Sair'):
            st.session_state['audit_log'].append((datetime.utcnow().isoformat(), st.session_state['user'], 'logout'))
            st.session_state['logged_in'] = False
            st.experimental_rerun()

# -----------------------------
# 1. Gerar / Carregar base sintética (5.000 clientes)
# -----------------------------
@st.cache_data
def generate_data(n=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    N = n
    client_names = [f'Cliente_{i:04d}' for i in range(1, N+1)]
    client_type = np.random.choice(['individual', 'corporate', 'vip'], N, p=[0.7, 0.25, 0.05])
    completeness = np.random.choice(['complete', 'partial', 'missing'], N, p=[0.6, 0.3, 0.1])
    frequency = np.random.choice(['recent', 'stale', 'never'], N, p=[0.5, 0.3, 0.2])
    pep = np.random.choice(['yes', 'no'], N, p=[0.06, 0.94])
    suspicious = np.random.choice(['low', 'medium', 'high'], N, p=[0.85, 0.12, 0.03])
    compliance_hist = np.random.choice(['clean', 'minor', 'major'], N, p=[0.9, 0.08, 0.02])
    income = np.random.choice(['low', 'medium', 'high'], N, p=[0.45, 0.4, 0.15])
    credit_hist = np.random.choice(['good', 'fair', 'bad', 'none'], N, p=[0.5, 0.25, 0.18, 0.07])
    guarantees = np.random.choice(['yes', 'no'], N, p=[0.3, 0.7])

    nationality = np.random.choice(['Cabo Verde', 'Portugal', 'Guiné-Bissau', 'Angola', 'Nigéria', 'Outros'], N, p=[0.6, 0.15, 0.1, 0.07, 0.05, 0.03])
    residence = np.random.choice(['Cabo Verde', 'Portugal', 'Luxemburgo', 'EUA', 'Angola'], N, p=[0.7, 0.1, 0.08, 0.07, 0.05])
    economic_activity = np.random.choice(['Comércio', 'Serviços', 'Construção', 'Turismo', 'Agricultura', 'Educação', 'Finanças'], N, p=[0.25,0.25,0.15,0.1,0.1,0.1,0.05])

    # Comerciais
    volume = np.round(np.abs(np.random.normal(20000, 50000, N))).astype(int)  # volume transações
    account_age = np.random.randint(0, 25, N)  # anos
    segment = np.random.choice(['Retalho', 'PME', 'Corporativo', 'Private'], N, p=[0.6,0.25,0.1,0.05])

    # Identificadores
    nif = [str(random.randint(1000000, 9999999)) for _ in range(N)]
    id_number = [f'ID{random.randint(10000,99999)}' for _ in range(N)]
    account_number = [f'AC{random.randint(100000,999999)}' for _ in range(N)]

    # Funções de risco com dependência de atividade económica
    def risk_operational(c, f, t, activity):
        score = 0
        if c=='missing': score += 2
        elif c=='partial': score += 1
        if f=='never': score += 2
        elif f=='stale': score += 1
        if t=='corporate': score += 1
        if activity in ['Construção','Turismo','Agricultura']: score += 1
        return 'high' if score>2 else ('medium' if score>0 else 'low')

    def risk_compliance(p, s, h):
        score = 0
        if p=='yes': score += 2
        if s=='high': score += 2
        elif s=='medium': score += 1
        if h=='major': score += 2
        elif h=='minor': score += 1
        return 'high' if score>3 else ('medium' if score>1 else 'low')

    def risk_credit(i, ch, g, vol):
        score = 0
        if i=='low': score += 2
        elif i=='medium': score += 1
        if ch=='bad': score += 2
        elif ch=='fair': score += 1
        if g=='no': score += 1
        if vol>50000: score -= 1
        return 'high' if score>3 else ('medium' if score>1 else 'low')

    operational = [risk_operational(c,f,t,a) for c,f,t,a in zip(completeness,frequency,client_type,economic_activity)]
    compliance = [risk_compliance(p,s,h) for p,s,h in zip(pep,suspicious,compliance_hist)]
    credit = [risk_credit(i,ch,g,v) for i,ch,g,v in zip(income,credit_hist,guarantees,volume)]

    # score final com ponderação
    def aggregate(op, comp, cr):
        score = 0
        score += {'low':0,'medium':2,'high':4}[op]
        score += {'low':0,'medium':2,'high':4}[comp]
        score += {'low':0,'medium':2,'high':4}[cr]
        if score<=3: return 'green'
        elif score<=7: return 'amber'
        else: return 'red'

    final = [aggregate(o,c,r) for o,c,r in zip(operational,compliance,credit)]

    df = pd.DataFrame({
        'Client_Name': client_names,
        'NIF': nif,
        'ID_Number': id_number,
        'Account_Number': account_number,
        'Nationality': nationality,
        'Country_Residence': residence,
        'Economic_Activity': economic_activity,
        'Volume_Transactions': volume,
        'Account_Age_Years': account_age,
        'Segment': segment,
        'Client_Type': client_type,
        'Completeness_Documents': completeness,
        'Frequency_Update': frequency,
        'PEP_Relationship': pep,
        'Suspicious_Transactions': suspicious,
        'Compliance_History': compliance_hist,
        'Income': income,
        'Credit_History': credit_hist,
        'Guarantees': guarantees,
        'Operational_Risk': operational,
        'Compliance_Risk': compliance,
        'Credit_Risk': credit,
        'Final_Score': final
    })
    return df

DATA = generate_data(5000)

# Guardar CSV para utilização offline
DATA.to_csv('clientes_banco_5000.csv', index=False, encoding='utf-8')

# -----------------------------
# 2. Construir e treinar a Rede Bayesiana (estrutura combinada)
# -----------------------------
EDGES = [
    ('Completeness_Documents', 'Operational_Risk'),
    ('Frequency_Update', 'Operational_Risk'),
    ('Client_Type', 'Operational_Risk'),
    ('PEP_Relationship', 'Compliance_Risk'),
    ('Suspicious_Transactions', 'Compliance_Risk'),
    ('Compliance_History', 'Compliance_Risk'),
    ('Operational_Risk', 'Final_Score'),
    ('Compliance_Risk', 'Final_Score'),
    ('Income', 'Credit_Risk'),
    ('Credit_History', 'Credit_Risk'),
    ('Guarantees', 'Credit_Risk'),
    ('Credit_Risk', 'Final_Score')
]
MODEL = DiscreteBayesianNetwork(EDGES)

# Ajuste (fit) com dados sintéticos (categorias já presentes)
try:
    MODEL.fit(DATA, estimator=BayesianEstimator, prior_type='BDeu')
except Exception:
    MODEL.fit(DATA, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(MODEL)

# -----------------------------
# 3. Funções utilitárias
# -----------------------------
def infer(evidence, nodes):
    q = {}
    for node in nodes:
        try:
            res = inference.query(variables=[node], evidence=evidence, show_progress=False)
            series = res[node]
            q[node] = {str(k): float(v) for k,v in series.items()}
        except Exception as e:
            q[node] = {'error': str(e)}
    # audit
    st.session_state['audit_log'].append((datetime.utcnow().isoformat(), st.session_state['user'], 'infer', evidence))
    return q

def network_fig(model):
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, seed=7)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0,y0 = pos[e[0]]
        x1,y1 = pos[e[1]]
        edge_x += [x0,x1,None]
        edge_y += [y0,y1,None]
    node_x, node_y = [], []
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x)
        node_y.append(y)
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition='top center', marker=dict(size=20))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title='Estrutura Rede Bayesiana', showlegend=False)
    return fig

# -----------------------------
# 4. Layout principal com abas
# -----------------------------
st.title('Portal Interno — Dashboard de Risco e Gestão Comercial')

tabs = st.tabs(['Visão Geral', 'Análise de Clientes', 'Rede Bayesiana', 'Alertas & Auditoria'])

# -----------------------------
# ABA 1: VISÃO GERAL (KPIs)
# -----------------------------
with tabs[0]:
    st.header('Visão Executiva')
    col1, col2, col3, col4 = st.columns(4)
    total_clients = len(DATA)
    kpi_risk_red = (DATA['Final_Score']=='red').mean()
    kpi_avg_volume = int(DATA['Volume_Transactions'].mean())
    kpi_pep = (DATA['PEP_Relationship']=='yes').sum()

    col1.metric('Clientes (Total)', f'{total_clients:,}')
    col2.metric('Percentual Alto Risco (red)', f'{kpi_risk_red*100:.2f}%')
    col3.metric('Volume Médio (transações)', f'{kpi_avg_volume:,}')
    col4.metric('Clientes PEP', f'{kpi_pep}')

    st.markdown('---')
    st.subheader('Distribuição de Risco por País de Residência')
    agg = DATA.groupby('Country_Residence')['Final_Score'].value_counts().unstack(fill_value=0)
    agg_norm = agg.div(agg.sum(axis=1), axis=0)
    fig = go.Figure()
    for col in agg_norm.columns:
        fig.add_trace(go.Bar(x=agg_norm.index, y=agg_norm[col], name=col))
    fig.update_layout(barmode='stack', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Top 8 Atividades por número de clientes e risco médio')
    top = DATA.groupby('Economic_Activity').agg({'Client_Name':'count','Final_Score':lambda s: (s.map({'green':0,'amber':1,'red':2}).mean())}).rename(columns={'Client_Name':'count','Final_Score':'risk_score'}).sort_values('count', ascending=False).head(8)
    fig2 = px.bar(top.reset_index(), x='Economic_Activity', y='count', color='risk_score', labels={'risk_score':'Risco médio (0=green,2=red)'})
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# ABA 2: ANÁLISE DE CLIENTES (BI) - filtros e tabela
# -----------------------------
with tabs[1]:
    st.header('Análise de Clientes — Filtros e Tabelas (Uso Interno)')
    with st.expander('Filtros rápidos'):
        c1, c2, c3, c4 = st.columns(4)
        country = c1.selectbox('País de Residência', options=['Todos']+sorted(DATA['Country_Residence'].unique().tolist()))
        activity = c2.selectbox('Atividade Económica', options=['Todas']+sorted(DATA['Economic_Activity'].unique().tolist()))
        segment = c3.selectbox('Segmento', options=['Todos']+sorted(DATA['Segment'].unique().tolist()))
        final_score = c4.selectbox('Score Final', options=['Todos']+sorted(DATA['Final_Score'].unique().tolist()))

    df_filtered = DATA.copy()
    if country!='Todos': df_filtered = df_filtered[df_filtered['Country_Residence']==country]
    if activity!='Todas': df_filtered = df_filtered[df_filtered['Economic_Activity']==activity]
    if segment!='Todos': df_filtered = df_filtered[df_filtered['Segment']==segment]
    if final_score!='Todos': df_filtered = df_filtered[df_filtered['Final_Score']==final_score]

    st.subheader(f'Resultado: {len(df_filtered):,} clientes')

    # Gráficos lado a lado
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('Distribuição por Risco Operacional')
        fig_op = px.histogram(df_filtered, x='Operational_Risk', title='Risco Operacional')
        st.plotly_chart(fig_op, use_container_width=True)
    with g2:
        st.markdown('Distribuição por Risco de Crédito')
        fig_cr = px.histogram(df_filtered, x='Credit_Risk', title='Risco de Crédito')
        st.plotly_chart(fig_cr, use_container_width=True)

    st.markdown('---')
    st.subheader('Tabela de Clientes (primeiras 100 linhas)')
    st.dataframe(df_filtered.head(100))

    # Export
    st.download_button('Exportar CSV (filtrado)', df_filtered.to_csv(index=False).encode('utf-8'), file_name='clientes_filtrados.csv')

# -----------------------------
# ABA 3: REDE BAYESIANA — SIMULAÇÃO E EXPLICAÇÃO
# -----------------------------
with tabs[2]:
    st.header('Rede Bayesiana — Simulação de Evidência (Uso para decisões operacionais)')
    st.subheader('Estrutura da Rede')
    st.plotly_chart(network_fig(MODEL), use_container_width=True)

    st.subheader('Inserir evidência (simular cliente)')
    e_client_type = st.selectbox('Tipo de cliente', options=sorted(DATA['Client_Type'].unique().tolist()))
    e_completeness = st.selectbox('Completude documentos', options=sorted(DATA['Completeness_Documents'].unique().tolist()))
    e_frequency = st.selectbox('Frequência atualização', options=sorted(DATA['Frequency_Update'].unique().tolist()))
    e_pep = st.selectbox('PEP?', options=sorted(DATA['PEP_Relationship'].unique().tolist()))
    e_suspicious = st.selectbox('Transações suspeitas', options=sorted(DATA['Suspicious_Transactions'].unique().tolist()))
    e_income = st.selectbox('Rendimento', options=sorted(DATA['Income'].unique().tolist()))
    e_credit_hist = st.selectbox('Histórico crédito', options=sorted(DATA['Credit_History'].unique().tolist()))
    e_guarantees = st.selectbox('Garantias', options=sorted(DATA['Guarantees'].unique().tolist()))

    evidence = {
        'Client_Type': e_client_type,
        'Completeness_Documents': e_completeness,
        'Frequency_Update': e_frequency,
        'PEP_Relationship': e_pep,
        'Suspicious_Transactions': e_suspicious,
        'Income': e_income,
        'Credit_History': e_credit_hist,
        'Guarantees': e_guarantees
    }

    if st.button('Executar inferência'):
        nodes = ['Operational_Risk','Compliance_Risk','Credit_Risk','Final_Score']
        result = infer(evidence, nodes)
        for node, probs in result.items():
            st.markdown(f'**{node}**')
            if 'error' in probs:
                st.error(probs['error'])
                continue
            fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs.keys()), orientation='h'))
            fig.update_layout(xaxis_title='Probabilidade', yaxis_title='Estado')
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ABA 4: ALERTAS & AUDITORIA
# -----------------------------
with tabs[3]:
    st.header('Alertas Operacionais e Registos de Auditoria')
    st.subheader('Regras de alerta (exemplos)')
    st.write('- Cliente PEP + transações suspeitas = alerta de compliance')
    st.write('- Documentos missing + volume alto = alerta operacional')
    st.write('- Score final = red => Escalar para gestor')

    # Gerar alertas simples a partir do dataset
    alerts = []
    sample = DATA.sample(2000, random_state=1)
    for _, row in sample.iterrows():
        if row['PEP_Relationship']=='yes' and row['Suspicious_Transactions']=='high':
            alerts.append({'Account':row['Account_Number'],'Reason':'PEP com transações altas','Client':row['Client_Name']})
        if row['Completeness_Documents']=='missing' and row['Volume_Transactions']>100000:
            alerts.append({'Account':row['Account_Number'],'Reason':'Documentos missing & volume elevado','Client':row['Client_Name']})
        if row['Final_Score']=='red':
            alerts.append({'Account':row['Account_Number'],'Reason':'Score final red','Client':row['Client_Name']})
    alerts_df = pd.DataFrame(alerts)
    st.subheader(f'Alertas detectados: {len(alerts_df)}')
    st.dataframe(alerts_df.head(200))

    st.markdown('---')
    st.subheader('Registos de auditoria (sessão)')
    audit_df = pd.DataFrame(st.session_state['audit_log'], columns=['timestamp','user','action','details'])
    # Alguns registos têm menos de 4 colunas — normalizar
    if not audit_df.empty:
        if audit_df.shape[1]==3:
            audit_df = pd.DataFrame([(r[0],r[1],r[2],None) for r in st.session_state['audit_log']], columns=['timestamp','user','action','details'])
    st.dataframe(audit_df.tail(200))

    # Allow managers to clear alerts (demo)
    if st.session_state['role']=='manager':
        if st.button('Limpar alertas (demo)'):
            st.success('Alertas limpos (simulação)')
            st.session_state['audit_log'].append((datetime.utcnow().isoformat(), st.session_state['user'], 'clear_alerts'))

# -----------------------------
# Notas finais e instruções de execução
# -----------------------------
st.sidebar.header('Instruções')
st.sidebar.markdown('''
- Ficheiro demo com autenticação simples (não usar em produção sem reforçar segurança).
- Para executar localmente: `streamlit run dashboard_banco_staff.py`
- Credenciais demo: analyst/analyst123 ; manager/manager123 ; auditor/auditor123
- O ficheiro `clientes_banco_5000.csv` é gerado automaticamente no diretório corrente.
- Para ligar a dados reais: substituir o bloco `generate_data` por `pd.read_csv()` e ajustar mapeamentos de categorias.
''')

# Fim do ficheiro
