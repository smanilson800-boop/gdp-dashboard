# -*- coding: utf-8 -*-
"""
================================================================================
Dashboard de Gestão de Risco Operacional - KYC
Trabalho de Fim de Curso - Matemática Aplicada
Universidade de Cabo Verde, 2026
Aluno: Manilson Semedo

Versão 6.5 - "Hard Rules Completas"
- PEP = Sim -> Risco Alto (regra determinística)
- Atividade de Risco -> Risco Alto (regra determinística)
- Não Residente + Portugal -> Risco Médio (regra determinística)
- Não Residente + País Africano -> Risco Alto (regra determinística)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO E CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="KYC Risk Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans Pro', sans-serif; }
    [data-testid="stSidebar"] { background-color: #F7F9FC; border-right: 1px solid #E1E8ED; }
    .kpi-card { background: #FFFFFF; border: 1px solid #E1E8ED; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 10px; }
    .kpi-title { font-size: 12px; color: #657786; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .kpi-value { font-size: 30px; font-weight: 700; color: #1F3A5F; margin-top: 5px; }
    .kpi-extra { font-size: 12px; color: #8899A6; margin-top: 5px; }
    .analise-box { background: #FFF9E6; border-left: 4px solid #F39C12; padding: 15px 20px; border-radius: 6px; margin: 20px 0; font-size: 14px; color: #2C3E50; line-height: 1.6; }
    .analise-box strong { color: #1F3A5F; }
    .analise-box.rule { background: #FDEDEC; border-left-color: #E74C3C; }
    .section-title { font-size: 26px; color: #1F3A5F; font-weight: 700; border-bottom: 3px solid #3498DB; padding-bottom: 10px; margin-bottom: 20px; }
    .section-subtitle { color: #657786; font-size: 15px; margin-bottom: 25px; font-style: italic; }
    .hard-rule-badge { display: inline-block; background: #E74C3C; color: white; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 5px; }
    .hard-rule-badge.orange { background: #F39C12; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
PAIS_PARA_ISO = {'Cabo Verde': 'CPV', 'Portugal': 'PRT', 'Angola': 'AGO', 'Brasil': 'BRA', 'United Kingdom': 'GBR', 'United States Of America': 'USA', 'France': 'FRA', 'Spain': 'ESP', 'Italy': 'ITA', 'Germany': 'DEU', 'Netherlands': 'NLD', 'Belgium': 'BEL', 'Switzerland': 'CHE', 'Ireland': 'IRL', 'Luxembourg': 'LUX', 'Sweden': 'SWE', 'Norway': 'NOR', 'Denmark': 'DNK', 'Poland': 'POL', 'Peoples Republic Of China': 'CHN', 'Japan': 'JPN', 'India': 'IND', 'Senegal': 'SEN', 'Guinea-Bissau': 'GNB', 'Guinea': 'GIN', 'Nigeria': 'NGA', 'Ghana': 'GHA', 'Mozambique': 'MOZ', 'Uganda': 'UGA', 'Kenya': 'KEN', 'South Africa': 'ZAF', 'Morocco': 'MAR', 'Tunisia': 'TUN', 'Pakistan': 'PAK', 'Saudi Arabia': 'SAU', 'Bahrain': 'BHR', 'United Arab Emirates': 'ARE', 'Australia': 'AUS', 'New Zealand': 'NZL', 'Canada': 'CAN', 'Mexico': 'MEX', 'Argentina': 'ARG', 'Congo Democratic Republic': 'COD', 'Equatorial Guinea': 'GNQ'}
FATF_NEGRO = ['COREA DO NORTE', 'IRAN', 'MYANMAR']
FATF_CINZA = ['ANGOLA', 'BOLIVIA', 'BULGARIA', 'CAMEROON', 'COTE D\'IVOIRE', 'CONGO, DEMOCRATIC REPUBLIC', 'HAITI', 'KENYA', 'KUWAIT', 'LEBANON', 'MONACO', 'NEPAL', 'PAPUA NEW GUINEA', 'SOUTH SUDAN', 'SYRIAN ARAB REPUBLIC', 'VENEZUELA', 'VIET NAM', 'YEMEN', 'CAMARÕES', 'COSTA DO MARFIM', 'REPÚBLICA DEMOCRÁTICA DO CONGO', 'QUÉNIA', 'LÍBANO', 'MÓNACO', 'PAPUA NOVA GUINÉ', 'SUDÃO DO SUL', 'SÍRIA', 'IÉMEN']

# Lista de países africanos (para a regra de Não Residente + África = Alto)
PAISES_AFRICANOS = [
    'ANGOLA', 'SENEGAL', 'GUINEA-BISSAU', 'GUINEA', 'NIGERIA', 'GHANA', 'MOZAMBIQUE', 
    'UGANDA', 'KENYA', 'SOUTH AFRICA', 'MOROCCO', 'TUNISIA', 'EGYPT', 'ALGERIA', 
    'CAMEROON', 'COTE D\'IVOIRE', 'CONGO', 'CONGO, DEMOCRATIC REPUBLIC', 
    'EQUATORIAL GUINEA', 'GABON', 'MALI', 'BURKINA FASO', 'NIGER', 'CHAD', 
    'SUDAN', 'SOUTH SUDAN', 'ETHIOPIA', 'SOMALIA', 'TANZANIA', 'ZAMBIA', 
    'ZIMBABWE', 'BOTSWANA', 'NAMIBIA', 'MADAGASCAR', 'MAURITIUS', 'RWANDA', 
    'BURUNDI', 'TOGO', 'BENIN', 'LIBERIA', 'SIERRA LEONE', 'GAMBIA', 
    'GAMBIA, THE', 'MAURITANIA', 'ERITREA', 'DJIBOUTI', 'LIBYA', 'CENTRAL AFRICAN REPUBLIC',
    'CAMARÕES', 'COSTA DO MARFIM', 'REPÚBLICA DEMOCRÁTICA DO CONGO', 'QUÉNIA', 
    'MOÇAMBIQUE', 'ÁFRICA DO SUL', 'MARROCOS', 'TUNÍSIA', 'EGIPTO', 'ARGÉLIA', 
    'SUDÃO DO SUL', 'ETIÓPIA', 'SOMÁLIA', 'TANZÂNIA', 'ZÂMBIA', 'ZIMBABUÉ', 
    'BOTSWANA', 'NAMÍBIA', 'MADAGÁSCAR', 'MAURÍCIA', 'RUANDA', 'BURUNDI', 
    'TOGO', 'BENIM', 'LIBÉRIA', 'SERRA LEOA', 'GÂMBIA', 'MAURITÂNIA', 
    'ERITREIA', 'JIBUTI', 'LÍBIA', 'REPÚBLICA CENTRO-AFRICANA'
]

# Atividades económicas de alto risco (regra de negócio)
ATIVIDADES_ALTO_RISCO = ['Atividades imobiliárias', 'Atividades financeiras e de seguros', 'Jogos e apostas']

CORES = {'Baixo': '#27AE60', 'Médio': '#F39C12', 'Alto': '#E74C3C'}
CORES_DOC = {'Válido': '#27AE60', 'Quase Expirando': '#F39C12', 'Expirado': '#E74C3C', 'Vitalício': '#3498DB'}

# ---------------------------------------------------------------------------
# FUNÇÕES DE REGRA GEOGRÁFICA
# ---------------------------------------------------------------------------
def is_pais_africano(pais):
    """Verifica se um país está na lista de países africanos."""
    if not pais:
        return False
    p = str(pais).upper().strip()
    return p in PAISES_AFRICANOS

def is_portugal(pais):
    """Verifica se é Portugal."""
    return str(pais).upper().strip() in ['PORTUGAL', 'PORTUGAL, REPÚBLICA PORTUGUESA']

def classificar_geo_risco(tipo_cliente, pais_residencia):
    """
    Retorna 'Médio', 'Alto' ou None (sem override geográfico).
    
    Regras:
    - Não Residente + Portugal -> Médio
    - Não Residente + País Africano -> Alto
    """
    if tipo_cliente != 'Não Residente':
        return None
    
    if is_portugal(pais_residencia):
        return 'Médio'
    
    if is_pais_africano(pais_residencia):
        return 'Alto'
    
    return None

# ---------------------------------------------------------------------------
# CARREGAMENTO
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar_dados():
    try:
        df = pd.read_csv('clientes_banco_real.csv')
    except FileNotFoundError:
        st.error("Ficheiro 'clientes_banco_real.csv' não encontrado.")
        st.stop()
        
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
    
    for col in ['Data_Validade', 'Data_Emissao']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if 'Data_Validade' in df.columns:
        df.loc[df['Data_Validade'].dt.year >= 2099, 'Status_Documento'] = 'Vitalício'
    
    if 'PEP_RCA' in df.columns:
        df['PEP_RCA'] = df['PEP_RCA'].apply(lambda x: 'Sim' if str(x).upper() in ['Y', 'S', 'SIM'] else 'Não')
    
    if 'Nivel_Risco' in df.columns:
        df['Nivel_Risco'] = df['Nivel_Risco'].replace({'LOW': 'Baixo', 'MED': 'Médio', 'HIGH': 'Alto'}).fillna('Médio')
    
    if 'Idade' in df.columns:
        df['Idade'] = pd.to_numeric(df['Idade'], errors='coerce')
        df['Faixa_Etaria'] = pd.cut(df['Idade'], bins=[0, 25, 40, 60, 75, 120], labels=['18-25', '26-40', '41-60', '61-75', '76+'])
    
    if 'Pais_Residencia' in df.columns:
        def classificar_fatf(pais):
            p = str(pais).upper().strip()
            if p in FATF_NEGRO: return 'Lista Negra'
            elif p in FATF_CINZA: return 'Lista Cinza'
            return 'Sem Sinalização'
        df['Pais_FATF'] = df['Pais_Residencia'].apply(classificar_fatf)
        df['Pais_Alto_Risco'] = df['Pais_FATF'].apply(lambda x: 'Sim' if x != 'Sem Sinalização' else 'Não')

    if 'Status_Documento' in df.columns:
        df['Documentacao_Expirada'] = df['Status_Documento'].apply(lambda x: 'Sim' if x == 'Expirado' else 'Não')

    if 'Tipo_Cliente' in df.columns:
        df['Cliente_Nao_Residente'] = df['Tipo_Cliente'].apply(lambda x: 'Sim' if x == 'Não Residente' else 'Não')

    # Mapeamento de Atividades para a Rede Bayesiana (3 grupos)
    if 'Atividade_Economica' in df.columns:
        def obter_grupo_atividade(ativ):
            ativ_str = str(ativ).strip()
            if any(r in ativ_str for r in ATIVIDADES_ALTO_RISCO):
                return 'Risco_Critico'
            elif 'construção' in ativ_str.lower():
                return 'Construcao'
            return 'Outros'
        df['Atividade_Grupo'] = df['Atividade_Economica'].apply(obter_grupo_atividade)
    
    return df

# ---------------------------------------------------------------------------
# REDE BAYESIANA COM HARD RULES (INCLUINDO GEOGRÁFICAS)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def criar_rede_bayesiana():
    """
    Rede Bayesiana com 5 evidências. As Hard Rules são aplicadas:
    1. Na CPD (para casos combinatórios)
    2. Na função de inferência em lote (para regras geográficas complexas)
    """
    modelo = DiscreteBayesianNetwork([
        ('Documentacao_Expirada', 'Nivel_Risco'),
        ('Cliente_Nao_Residente', 'Nivel_Risco'),
        ('PEP_RCA', 'Nivel_Risco'),
        ('Pais_Alto_Risco', 'Nivel_Risco'),
        ('Atividade_Grupo', 'Nivel_Risco'),
    ])

    estados_atividade = ['Risco_Critico', 'Construcao', 'Outros'] 
    
    cpds = [
        TabularCPD('Documentacao_Expirada', 2, [[0.85], [0.15]], state_names={'Documentacao_Expirada': ['Não', 'Sim']}),
        TabularCPD('Cliente_Nao_Residente', 2, [[0.85], [0.15]], state_names={'Cliente_Nao_Residente': ['Não', 'Sim']}),
        TabularCPD('PEP_RCA', 2, [[0.97], [0.03]], state_names={'PEP_RCA': ['Não', 'Sim']}),
        TabularCPD('Pais_Alto_Risco', 2, [[0.98], [0.02]], state_names={'Pais_Alto_Risco': ['Não', 'Sim']}),
        TabularCPD('Atividade_Grupo', 3, [[0.25], [0.20], [0.55]], state_names={'Atividade_Grupo': estados_atividade}),
    ]

    valores = []
    for doc in [0, 1]:
        for nao_res in [0, 1]:
            for pep in [0, 1]:
                for fatf in [0, 1]:
                    for ativ_idx in [0, 1, 2]:
                        # =====================================================
                        # HARD RULES NA CPD
                        # =====================================================
                        
                        # Regra 1: PEP = Sim -> SEMPRE Alto Risco (100%)
                        if pep == 1:
                            probs = [0.0, 0.0, 1.0]
                        
                        # Regra 2: Atividade de Risco Crítico -> SEMPRE Alto Risco (100%)
                        elif ativ_idx == 0:
                            probs = [0.0, 0.0, 1.0]
                        
                        # Regra 3: Casos normais (Matemática probabilística)
                        else:
                            score = (doc * 3 + nao_res * 3 + fatf * 4)
                            if ativ_idx == 1: score += 1
                            
                            if score <= 2: probs = [0.82, 0.14, 0.04]
                            elif score <= 5: probs = [0.42, 0.44, 0.14]
                            elif score <= 8: probs = [0.15, 0.55, 0.30]
                            else: probs = [0.04, 0.18, 0.78]
                            
                        valores.append(probs)

    cpds.append(TabularCPD(
        'Nivel_Risco', 3, np.array(valores).T.tolist(),
        evidence=['Documentacao_Expirada', 'Cliente_Nao_Residente', 'PEP_RCA', 'Pais_Alto_Risco', 'Atividade_Grupo'],
        evidence_card=[2, 2, 2, 2, 3],
        state_names={
            'Nivel_Risco': ['Baixo', 'Médio', 'Alto'],
            'Documentacao_Expirada': ['Não', 'Sim'],
            'Cliente_Nao_Residente': ['Não', 'Sim'],
            'PEP_RCA': ['Não', 'Sim'],
            'Pais_Alto_Risco': ['Não', 'Sim'],
            'Atividade_Grupo': estados_atividade,
        }))

    modelo.add_cpds(*cpds)
    if modelo.check_model():
        return modelo, VariableElimination(modelo)
    return None, None


def inferir_risco(inferencia, evidencias):
    try:
        ev = {k: v for k, v in evidencias.items() if v not in ('Não informado', None, '')}
        if not ev: return None
        resultado = inferencia.query(variables=['Nivel_Risco'], evidence=ev)
        return {estado: float(resultado.values[i]) for i, estado in enumerate(resultado.state_names['Nivel_Risco'])}
    except Exception:
        return None


def aplicar_hard_rules_completas(prob, linha):
    """
    Aplica TODAS as Hard Rules à probabilidade calculada.
    Devolve a probabilidade final após override.
    """
    prob_final = prob.copy()
    
    # --- HARD RULE 1: PEP = Sim -> Alto Risco ---
    if linha.get('PEP_RCA') == 'Sim':
        prob_final = {'Baixo': 0.0, 'Médio': 0.0, 'Alto': 1.0}
        return prob_final
    
    # --- HARD RULE 2: Atividade de Risco -> Alto Risco ---
    if linha.get('Atividade_Grupo') == 'Risco_Critico':
        prob_final = {'Baixo': 0.0, 'Médio': 0.0, 'Alto': 1.0}
        return prob_final
    
    # --- HARD RULE 3 e 4: Regras Geográficas (Não Residente) ---
    if linha.get('Cliente_Nao_Residente') == 'Sim':
        pais = linha.get('Pais_Residencia', '')
        
        if is_portugal(pais):
            # Não Residente + Portugal -> Médio
            prob_final = {'Baixo': 0.0, 'Médio': 1.0, 'Alto': 0.0}
            return prob_final
        
        if is_pais_africano(pais):
            # Não Residente + África -> Alto
            prob_final = {'Baixo': 0.0, 'Médio': 0.0, 'Alto': 1.0}
            return prob_final
    
    return prob_final


def aplicar_rede_em_lote(df, inferencia):
    """
    Aplica a rede a toda a base, com HARD RULES completas.
    """
    resultados = []
    for _, linha in df.iterrows():
        evidencias = {
            'Documentacao_Expirada': linha.get('Documentacao_Expirada', 'Não'),
            'Cliente_Nao_Residente': linha.get('Cliente_Nao_Residente', 'Não'),
            'PEP_RCA': linha.get('PEP_RCA', 'Não'),
            'Pais_Alto_Risco': linha.get('Pais_Alto_Risco', 'Não'),
            'Atividade_Grupo': linha.get('Atividade_Grupo', 'Outros'),
        }
        prob = inferir_risco(inferencia, evidencias)
        if prob:
            # Aplicar TODAS as Hard Rules
            prob_final = aplicar_hard_rules_completas(prob, linha)
            classe = max(prob_final, key=prob_final.get)
            
            resultados.append({
                'Classe_RB': classe,
                'Classe_AML': linha.get('Nivel_Risco', 'Médio'),
                'Concorda': classe == linha.get('Nivel_Risco', 'Médio'),
                'Prob_Alto': prob_final.get('Alto', 0),
            })
    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# INICIALIZAÇÃO
# ---------------------------------------------------------------------------
df = carregar_dados()

with st.sidebar:
    st.markdown("# 📊 KYC Risk Analytics")
    st.markdown("---")
    pagina = st.radio("Navegação", [
        "🏠 Visão Geral",
        "📈 Risco & Distribuição",
        "📄 Documentação",
        "👥 Perfil Demográfico",
        "🌍 Análise Geográfica",
        "🎯 Segmentação de Risco",
        "🧠 Rede Bayesiana"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**Filtros Globais**")
    filtro_risco = st.multiselect("Nível de Risco", ['Baixo', 'Médio', 'Alto'], default=['Baixo', 'Médio', 'Alto'])
    filtro_tipo = st.multiselect("Tipo de Cliente", df['Tipo_Cliente'].dropna().unique().tolist(), default=df['Tipo_Cliente'].dropna().unique().tolist())

df_f = df[df['Nivel_Risco'].isin(filtro_risco) & df['Tipo_Cliente'].isin(filtro_tipo)].copy()

if df_f.empty:
    st.warning("Nenhum cliente corresponde aos filtros.")
    st.stop()

# ---------------------------------------------------------------------------
# PÁGINAS
# ---------------------------------------------------------------------------
if pagina == "🏠 Visão Geral":
    st.markdown('<div class="section-title">Visão Geral da Carteira</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Panorama global dos indicadores de risco operacional</div>', unsafe_allow_html=True)
    
    total = len(df_f)
    pct_alto = (df_f['Nivel_Risco'] == 'Alto').mean() * 100
    n_doc_exp = (df_f['Status_Documento'] == 'Expirado').sum()
    n_pep = (df_f['PEP_RCA'] == 'Sim').sum()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total de Clientes</div><div class="kpi-value">{total:,}</div><div class="kpi-extra">carteira analisada</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Risco Alto</div><div class="kpi-value">{pct_alto:.1f}%</div><div class="kpi-extra">{int(total*pct_alto/100):,} clientes</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Docs Expirados</div><div class="kpi-value">{n_doc_exp:,}</div><div class="kpi-extra">{n_doc_exp/total*100:.1f}% da carteira</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">PEP/RCA</div><div class="kpi-value">{n_pep:,}</div><div class="kpi-extra">{n_pep/total*100:.1f}% da carteira</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribuição Global de Risco**")
        contagem = df_f['Nivel_Risco'].value_counts().reset_index()
        contagem.columns = ['Risco', 'N']
        fig1 = go.Figure(data=[go.Pie(labels=contagem['Risco'], values=contagem['N'], hole=0.6, marker=dict(colors=[CORES[r] for r in contagem['Risco']], line=dict(color='white', width=3)), textinfo='percent+label', textposition='outside')])
        fig1.update_layout(height=380, showlegend=False, margin=dict(t=30, b=30))
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.markdown("**Composição por Tipo de Cliente**")
        contagem_tipo = df_f['Tipo_Cliente'].value_counts().reset_index()
        contagem_tipo.columns = ['Tipo', 'N']
        fig2 = px.bar(contagem_tipo, x='Tipo', y='N', color='Tipo', text='N', color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_title="Nº de Clientes", plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown(f'<div class="analise-box"><strong>📝 Análise:</strong> A carteira tem <strong>{total:,} clientes</strong>, com <strong>{pct_alto:.1f}%</strong> em risco alto. A maioria ({df_f["Tipo_Cliente"].value_counts().index[0]}) concentra {df_f["Tipo_Cliente"].value_counts().values[0]/total*100:.1f}% da carteira.</div>', unsafe_allow_html=True)

elif pagina == "📈 Risco & Distribuição":
    st.markdown('<div class="section-title">Análise de Distribuição de Risco</div>', unsafe_allow_html=True)
    st.markdown("**Risco por Tipo de Cliente**")
    tabela1 = pd.crosstab(df_f['Tipo_Cliente'], df_f['Nivel_Risco'])
    for r in ['Baixo', 'Médio', 'Alto']:
        if r not in tabela1.columns: tabela1[r] = 0
    tabela1 = tabela1[['Baixo', 'Médio', 'Alto']]
    fig3 = go.Figure()
    for risco in ['Baixo', 'Médio', 'Alto']:
        fig3.add_trace(go.Bar(name=risco, x=tabela1.index, y=tabela1[risco], marker_color=CORES[risco], text=tabela1[risco].apply(lambda v: f'{v:,}' if v > 0 else ''), textposition='inside', textfont=dict(color='white', size=11)))
    fig3.update_layout(barmode='stack', height=400, plot_bgcolor='white', xaxis_title="Tipo de Cliente", yaxis_title="Nº de Clientes", legend_title="Nível de Risco", margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribuição de Risco por Faixa Etária**")
        if 'Faixa_Etaria' in df_f.columns:
            tabela2 = pd.crosstab(df_f['Faixa_Etaria'], df_f['Nivel_Risco'])
            for r in ['Baixo', 'Médio', 'Alto']:
                if r not in tabela2.columns: tabela2[r] = 0
            tabela2 = tabela2[['Baixo', 'Médio', 'Alto']]
            fig4 = go.Figure()
            for risco in ['Baixo', 'Médio', 'Alto']:
                fig4.add_trace(go.Bar(name=risco, x=tabela2.index, y=tabela2[risco], marker_color=CORES[risco]))
            fig4.update_layout(barmode='stack', height=380, plot_bgcolor='white', xaxis_title="Faixa Etária", yaxis_title="Nº de Clientes", showlegend=False, margin=dict(t=20))
            st.plotly_chart(fig4, use_container_width=True)
    with c2:
        st.markdown("**Risco por Sexo**")
        tabela3 = pd.crosstab(df_f['Sexo'], df_f['Nivel_Risco'])
        for r in ['Baixo', 'Médio', 'Alto']:
            if r not in tabela3.columns: tabela3[r] = 0
        tabela3 = tabela3[['Baixo', 'Médio', 'Alto']]
        fig5 = go.Figure()
        for risco in ['Baixo', 'Médio', 'Alto']:
            fig5.add_trace(go.Bar(name=risco, x=tabela3.index, y=tabela3[risco], marker_color=CORES[risco]))
        fig5.update_layout(barmode='group', height=380, plot_bgcolor='white', xaxis_title="Sexo", yaxis_title="Nº de Clientes", legend_title="Risco", margin=dict(t=20))
        st.plotly_chart(fig5, use_container_width=True)

elif pagina == "📄 Documentação":
    st.markdown('<div class="section-title">Gestão Documental</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Válidos</div><div class="kpi-value">{(df_f["Status_Documento"]=="Válido").sum():,}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Quase Expirando</div><div class="kpi-value">{(df_f["Status_Documento"]=="Quase Expirando").sum():,}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Expirados</div><div class="kpi-value">{(df_f["Status_Documento"]=="Expirado").sum():,}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Vitalícios</div><div class="kpi-value">{(df_f["Status_Documento"]=="Vitalício").sum():,}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        contagem = df_f['Status_Documento'].value_counts().reset_index()
        contagem.columns = ['Status', 'N']
        fig6 = go.Figure(data=[go.Pie(labels=contagem['Status'], values=contagem['N'], hole=0.55, marker=dict(colors=[CORES_DOC.get(s, '#95A5A6') for s in contagem['Status']], line=dict(color='white', width=3)), textinfo='percent', textposition='outside')])
        fig6.update_layout(height=400, showlegend=True, legend=dict(orientation='h', y=-0.2), margin=dict(t=20))
        st.plotly_chart(fig6, use_container_width=True)
    with c2:
        df_exp = df_f[df_f['Status_Documento'] == 'Expirado']
        if not df_exp.empty:
            contagem_exp = df_exp['Tipo_Cliente'].value_counts().reset_index()
            contagem_exp.columns = ['Tipo', 'N']
            fig7 = px.bar(contagem_exp, x='N', y='Tipo', orientation='h', color='N', color_continuous_scale='Reds', text='N')
            fig7.update_traces(textposition='outside')
            fig7.update_layout(height=400, plot_bgcolor='white', xaxis_title="Nº de Documentos Expirados", yaxis_title="", coloraxis_showscale=False, margin=dict(t=20))
            st.plotly_chart(fig7, use_container_width=True)

elif pagina == "👥 Perfil Demográfico":
    st.markdown('<div class="section-title">Perfil Demográfico</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Idade Média</div><div class="kpi-value">{df_f["Idade"].mean():.0f} anos</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Cliente Mais Velho</div><div class="kpi-value">{df_f["Idade"].max():.0f} anos</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Mulheres</div><div class="kpi-value">{(df_f["Sexo"]=="Feminino").mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if 'Faixa_Etaria' in df_f.columns:
            tabela = pd.crosstab(df_f['Faixa_Etaria'], df_f['Nivel_Risco'])
            for r in ['Baixo', 'Médio', 'Alto']:
                if r not in tabela.columns: tabela[r] = 0
            tabela = tabela[['Baixo', 'Médio', 'Alto']]
            fig10 = go.Figure()
            for risco in ['Baixo', 'Médio', 'Alto']:
                fig10.add_trace(go.Bar(name=risco, x=tabela.index, y=tabela[risco], marker_color=CORES[risco]))
            fig10.update_layout(barmode='stack', height=380, plot_bgcolor='white', xaxis_title="Faixa Etária", yaxis_title="Nº de Clientes", showlegend=False, margin=dict(t=20))
            st.plotly_chart(fig10, use_container_width=True)
    with c2:
        contagem_ativ = df_f['Atividade_Economica'].value_counts().head(10).reset_index()
        contagem_ativ.columns = ['Atividade', 'N']
        fig11 = px.bar(contagem_ativ, x='N', y='Atividade', orientation='h', text='N', color='N', color_continuous_scale='Teal')
        fig11.update_traces(textposition='outside')
        fig11.update_layout(height=380, plot_bgcolor='white', xaxis_title="", yaxis_title="", coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'}, margin=dict(t=20))
        st.plotly_chart(fig11, use_container_width=True)

elif pagina == "🌍 Análise Geográfica":
    st.markdown('<div class="section-title">Análise Geográfica do Risco</div>', unsafe_allow_html=True)
    paises_agg = df_f.groupby('Pais_Residencia').agg(Total=('ID_Cliente', 'count'), Pct_Alto=('Nivel_Risco', lambda x: (x == 'Alto').mean() * 100)).reset_index()
    paises_agg['ISO'] = paises_agg['Pais_Residencia'].map(PAIS_PARA_ISO)
    paises_agg = paises_agg.dropna(subset=['ISO'])
    if not paises_agg.empty:
        fig_map = px.choropleth(paises_agg, locations='ISO', color='Pct_Alto', hover_name='Pais_Residencia', hover_data={'Total': True, 'Pct_Alto': ':.1f'}, color_continuous_scale='RdYlGn_r', range_color=[0, 100], labels={'Pct_Alto': '% Risco Alto', 'Total': 'Clientes'})
        fig_map.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True, coastlinecolor='#ccc', projection_type='natural earth'), margin=dict(t=20, b=20, l=0, r=0), coloraxis_colorbar=dict(title="% Risco"))
        st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("---")
    st.markdown("**Top 15 Países por Número de Clientes**")
    top_paises = df_f['Pais_Residencia'].value_counts().head(15).reset_index()
    top_paises.columns = ['País', 'N']
    fig12 = px.bar(top_paises, x='N', y='País', orientation='h', text='N', color='N', color_continuous_scale='Blues')
    fig12.update_traces(textposition='outside')
    fig12.update_layout(height=500, plot_bgcolor='white', xaxis_title="Nº de Clientes", yaxis_title="", coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'}, margin=dict(t=20))
    st.plotly_chart(fig12, use_container_width=True)

elif pagina == "🎯 Segmentação de Risco":
    st.markdown('<div class="section-title">Segmentação Avançada de Risco</div>', unsafe_allow_html=True)
    st.markdown("**Mapa de Calor: Risco por Tipo de Cliente e PEP/RCA**")
    if 'PEP_RCA' in df_f.columns:
        df_f['Categoria'] = df_f['Tipo_Cliente'] + ' / ' + df_f['PEP_RCA']
        tabela = pd.crosstab(df_f['Categoria'], df_f['Nivel_Risco'])
        for r in ['Baixo', 'Médio', 'Alto']:
            if r not in tabela.columns: tabela[r] = 0
        tabela = tabela[['Baixo', 'Médio', 'Alto']]
        tabela['_Total'] = tabela.sum(axis=1)
        tabela = tabela[tabela['_Total'] >= 5]
        tabela = tabela.drop(columns='_Total').sort_values('Alto', ascending=False).head(15)
        fig15 = go.Figure(data=go.Heatmap(z=tabela.values, x=tabela.columns, y=tabela.index, colorscale=[[0, '#EBF5FB'], [0.5, '#F39C12'], [1, '#E74C3C']], text=tabela.values, texttemplate="%{text}", textfont=dict(size=11, color='black'), showscale=True, colorbar=dict(title="Nº")))
        fig15.update_layout(height=500, plot_bgcolor='white', xaxis_title="Nível de Risco", yaxis_title="", margin=dict(t=20, l=150))
        st.plotly_chart(fig15, use_container_width=True)

# ---------------------------------------------------------------------------
# PÁGINA 7: REDE BAYESIANA (COM TODAS AS HARD RULES)
# ---------------------------------------------------------------------------
elif pagina == "🧠 Rede Bayesiana":
    st.markdown('<div class="section-title">Modelação Probabilística (Rede Bayesiana)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Inferência com regras determinísticas de negócio (Hard Rules)</div>', unsafe_allow_html=True)
    
    modelo, inferencia = criar_rede_bayesiana()
    
    if modelo is None:
        st.error("Erro na construção do modelo Bayesiano.")
        st.stop()
    
    # --- CAIXA DE HARD RULES ---
    st.markdown(f"""
    <div class="analise-box rule">
        <strong>⚠️ REGRAS DETERMINÍSTICAS (HARD RULES) DO MODELO:</strong><br>
        <span class="hard-rule-badge">REGRA 1</span> Se <strong>PEP/RCA = Sim</strong> → <strong>Risco Alto</strong> (100%).<br>
        <span class="hard-rule-badge">REGRA 2</span> Se <strong>Atividade = Imobiliária/Financeira/Jogo</strong> → <strong>Risco Alto</strong> (100%).<br>
        <span class="hard-rule-badge orange">REGRA 3</span> Se <strong>Não Residente + Portugal</strong> → <strong>Risco Médio</strong> (regra geográfica).<br>
        <span class="hard-rule-badge">REGRA 4</span> Se <strong>Não Residente + País Africano</strong> → <strong>Risco Alto</strong> (regra geográfica).<br>
        <br>
        Estas regras <strong>sobrepõem-se</strong> à inferência probabilística. A presença de PEP ou atividade de risco crítico força sempre Alto Risco. As regras geográficas aplicam-se apenas a Não Residentes.
    </div>
    """, unsafe_allow_html=True)
    
    # --- GRÁFICO DO GRAFO (DAG) ---
    st.markdown("**Estrutura do Grafo (DAG)**")
    G = nx.DiGraph(modelo.edges())
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        node_x.append(pos[node][0]); node_y.append(pos[node][1])
        node_text.append(node.replace('_', ' '))
        node_color.append('#F39C12' if node == 'Nivel_Risco' else '#1F3A5F')
        node_size.append(80 if node == 'Nivel_Risco' else 55)
    fig_rb = go.Figure()
    fig_rb.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#B0BEC5'), mode='lines', hoverinfo='none'))
    fig_rb.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center', textfont=dict(size=11, color='#1F3A5F'), marker=dict(size=node_size, color=node_color, line=dict(width=3, color='white'))))
    fig_rb.update_layout(height=500, plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(t=40, b=20, l=20, r=20), showlegend=False)
    st.plotly_chart(fig_rb, use_container_width=True)
    
    st.markdown("---")
    
    # --- SIMULADOR INTERATIVO ---
    st.markdown("**Simulador de Cenários (Inferência Preditiva)**")
    st.caption("Experimente as regras: PEP=Sim → Alto; Atividade Imobiliária → Alto; Não Residente + Portugal → Médio; Não Residente + África → Alto.")

    lista_todas_atividades = sorted(df['Atividade_Economica'].dropna().astype(str).unique().tolist())
    lista_paises = sorted(df['Pais_Residencia'].dropna().astype(str).unique().tolist())
    
    c1, c2, c3 = st.columns(3)
    opcoes_binarias = ['Não informado', 'Não', 'Sim']

    with c1:
        ev_doc = st.selectbox("Documentação Expirada", opcoes_binarias, key='sim_doc')
        ev_naores = st.selectbox("Cliente Não Residente", opcoes_binarias, key='sim_naores')
    with c2:
        ev_pep = st.selectbox("PEP/RCA", opcoes_binarias, key='sim_pep')
        ev_pais = st.selectbox("País de Residência", ['Não informado'] + lista_paises, key='sim_pais')
    with c3:
        ev_ativ = st.selectbox("Atividade Económica", ['Não informado'] + lista_todas_atividades, key='sim_ativ')

    if st.button("Calcular Probabilidades", type="primary"):
        evidencias = {}
        if ev_doc != 'Não informado': evidencias['Documentacao_Expirada'] = ev_doc
        if ev_naores != 'Não informado': evidencias['Cliente_Nao_Residente'] = ev_naores
        if ev_pep != 'Não informado': evidencias['PEP_RCA'] = ev_pep
        if ev_pais != 'Não informado':
            p_upper = ev_pais.upper().strip()
            evidencias['Pais_Alto_Risco'] = 'Sim' if (p_upper in FATF_NEGRO or p_upper in FATF_CINZA) else 'Não'
        if ev_ativ != 'Não informado':
            if any(r in ev_ativ for r in ATIVIDADES_ALTO_RISCO): 
                evidencias['Atividade_Grupo'] = 'Risco_Critico'
            elif 'construção' in ev_ativ.lower(): 
                evidencias['Atividade_Grupo'] = 'Construcao'
            else: 
                evidencias['Atividade_Grupo'] = 'Outros'

        # 1. Calcular probabilidade base (Bayes)
        probabilidades = inferir_risco(inferencia, evidencias)
        
        if probabilidades:
            # 2. Aplicar Hard Rules completas (incluindo geográficas)
            linha_simulada = {
                'PEP_RCA': ev_pep,
                'Atividade_Grupo': evidencias.get('Atividade_Grupo', 'Outros'),
                'Cliente_Nao_Residente': ev_naores,
                'Pais_Residencia': ev_pais if ev_pais != 'Não informado' else ''
            }
            prob_final = aplicar_hard_rules_completas(probabilidades, linha_simulada)
            
            # 3. Mostrar se houve override
            houve_override = prob_final != probabilidades
            if houve_override:
                st.markdown("""
                <div style="background: #FDEDEC; border-left: 4px solid #E74C3C; padding: 10px 15px; border-radius: 6px; margin-bottom: 15px;">
                    <strong>⚠️ HARD RULE APLICADA:</strong> A probabilidade base foi sobreposta por uma regra determinística de negócio.
                </div>
                """, unsafe_allow_html=True)
            
            # 4. Mostrar probabilidades finais
            c1, c2, c3 = st.columns(3)
            for col, risco in zip([c1, c2, c3], ['Baixo', 'Médio', 'Alto']):
                with col:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-title">P(Risco = {risco})</div><div class="kpi-value" style="color:{CORES[risco]}">{prob_final[risco]:.1%}</div></div>', unsafe_allow_html=True)
            
            fig_prob = go.Figure(go.Bar(x=['Baixo', 'Médio', 'Alto'], y=[prob_final[r] for r in ['Baixo', 'Médio', 'Alto']], marker_color=[CORES[r] for r in ['Baixo', 'Médio', 'Alto']], text=[f"{prob_final[r]:.1%}" for r in ['Baixo', 'Médio', 'Alto']], textposition='outside'))
            fig_prob.update_layout(title='Distribuição de Probabilidade Final (após Hard Rules)', yaxis_range=[0, 1.15], height=400, plot_bgcolor='white')
            st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")
    
    # --- VALIDAÇÃO EM LOTE ---
    st.markdown("**Validação em Lote (Rede Bayesiana vs AMLRISK)**")
    if st.button("Aplicar Modelo a Todos os Clientes", type="primary"):
        with st.spinner("A processar inferência..."):
            df_val = aplicar_rede_em_lote(df_f, inferencia)
            st.session_state['df_val'] = df_val
            st.success(f"Concluído: {len(df_val):,} clientes.")

    if 'df_val' in st.session_state:
        df_val = st.session_state['df_val']
        n = len(df_val)
        if n > 0:
            n_ok = df_val['Concorda'].sum()
            taxa = n_ok / n * 100
            kappa = cohen_kappa_score(df_val['Classe_AML'], df_val['Classe_RB'])
            acuracia = accuracy_score(df_val['Classe_AML'], df_val['Classe_RB'])
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Concordância</div><div class="kpi-value">{taxa:.1f}%</div><div class="kpi-extra">{n_ok}/{n}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Cohen Kappa</div><div class="kpi-value">{kappa:.3f}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-title">Acurácia</div><div class="kpi-value">{acuracia:.1%}</div></div>', unsafe_allow_html=True)

            cm = confusion_matrix(df_val['Classe_AML'], df_val['Classe_RB'], labels=['Baixo', 'Médio', 'Alto'])
            c1, c2 = st.columns(2)
            with c1:
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=[f"RB: {r}" for r in ['Baixo', 'Médio', 'Alto']], y=[f"AML: {r}" for r in ['Baixo', 'Médio', 'Alto']], text=cm, texttemplate="%{text}", colorscale='Blues', showscale=False, textfont=dict(size=14, color='white')))
                fig_cm.update_layout(title='Matriz de Confusão', height=400)
                st.plotly_chart(fig_cm, use_container_width=True)
            with c2:
                comparacao = pd.DataFrame({'Classe': ['Baixo', 'Médio', 'Alto'] * 2, 'Modelo': ['AMLRISK'] * 3 + ['Bayes'] * 3, 'Contagem': [(df_val['Classe_AML'] == r).sum() for r in ['Baixo', 'Médio', 'Alto']] + [(df_val['Classe_RB'] == r).sum() for r in ['Baixo', 'Médio', 'Alto']]})
                fig_comp = px.bar(comparacao, x='Classe', y='Contagem', color='Modelo', barmode='group', color_discrete_map={'AMLRISK': '#3498DB', 'Bayes': '#F39C12'}, text='Contagem')
                fig_comp.update_traces(textposition='outside')
                fig_comp.update_layout(title='Comparação de Contagens', height=400, plot_bgcolor='white')
                st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown(f'<div class="analise-box"><strong>📝 Análise:</strong> A rede obteve <strong>{taxa:.1f}%</strong> de concordância. Kappa = <strong>{kappa:.3f}</strong>. As 4 Hard Rules (PEP, Atividade de Risco, Portugal → Médio, África → Alto) foram aplicadas com sucesso.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #8899A6; font-size: 12px; padding: 20px;'>Dashboard desenvolvido por Manilson Semedo | TFC Matemática Aplicada | Uni-CV 2026</div>", unsafe_allow_html=True)