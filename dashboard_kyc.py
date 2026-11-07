"""
=============================================================================
DASHBOARD INTERATIVO - SISTEMA DE AVALIAÇÃO DE RISCO KYC
COM REDES BAYESIANAS NO CONTEXTO DE CABO VERDE
=============================================================================
Autor: [Seu Nome]
TCC: Implementação de um Sistema de Avaliação de Risco KYC com Redes Bayesianas
Instituição: [Sua Instituição] - Cabo Verde
Data: Novembro 2026
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="KYC Risk Assessment - Cabo Verde",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ESTILO CSS PERSONALIZADO
# =============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003366 0%, #0066cc 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .risk-alto { border-left-color: #dc3545 !important; }
    .risk-medio { border-left-color: #ffc107 !important; }
    .risk-baixo { border-left-color: #28a745 !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARREGAMENTO E PROCESSAMENTO DE DADOS
# =============================================================================
@st.cache_data
def carregar_dados():
    """Carrega e processa a base de dados de clientes KYC"""
    df = pd.read_csv('clientes_banco_real.csv')
    
    # Converter datas
    df['Data_Validade'] = pd.to_datetime(df['Data_Validade'], errors='coerce')
    df['Data_Emissao'] = pd.to_datetime(df['Data_Emissao'], errors='coerce')
    
    # Criar categorias de idade
    bins_idade = [0, 25, 35, 45, 55, 65, 100]
    labels_idade = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['Faixa_Etaria'] = pd.cut(df['Idade'], bins=bins_idade, labels=labels_idade, right=True)
    
    # Criar categorias de tempo de relacionamento
    bins_tempo = [0, 2, 5, 10, 15, 25, 30]
    labels_tempo = ['0-2 anos', '3-5 anos', '6-10 anos', '11-15 anos', '16-25 anos', '25+ anos']
    df['Faixa_Tempo_Relacionamento'] = pd.cut(
        df['Tempo_Relacionamento'], 
        bins=bins_tempo, 
        labels=labels_tempo, 
        right=True
    )
    
    return df

# =============================================================================
# FUNÇÕES AUXILIARES PARA ANÁLISE ESTATÍSTICA
# =============================================================================
def cramers_v(x, y):
    """Calcula o V de Cramer para medir associação entre variáveis categóricas"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def calcular_matriz_associacao(df, colunas):
    """Calcula matriz de associação (V de Cramer) entre variáveis categóricas"""
    n = len(colunas)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matriz[i, j] = 1.0
            else:
                try:
                    matriz[i, j] = cramers_v(df[colunas[i]], df[colunas[j]])
                except:
                    matriz[i, j] = 0.0
    return pd.DataFrame(matriz, index=colunas, columns=colunas)

# =============================================================================
# CARREGAR DADOS
# =============================================================================
df = carregar_dados()

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🏦 Sistema de Avaliação de Risco KYC</h1>
    <h3>Implementação com Redes Bayesianas no Contexto de Cabo Verde</h3>
    <p>Dashboard Interativo para Trabalho de Conclusão de Curso</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - FILTROS
# =============================================================================
st.sidebar.header("🔍 Filtros de Análise")

# Filtro de Nível de Risco
nivel_risco = st.sidebar.multiselect(
    "Nível de Risco:",
    options=df['Nivel_Risco'].unique(),
    default=df['Nivel_Risco'].unique()
)

# Filtro de Tipo de Cliente
tipo_cliente = st.sidebar.multiselect(
    "Tipo de Cliente:",
    options=df['Tipo_Cliente'].unique(),
    default=df['Tipo_Cliente'].unique()
)

# Filtro de PEP
pep_filter = st.sidebar.selectbox(
    "Pessoa Exposta Politicamente (PEP):",
    options=["Todos", "Sim", "Não"]
)

# Filtro de Status do Documento
status_doc = st.sidebar.multiselect(
    "Status do Documento:",
    options=df['Status_Documento'].unique(),
    default=df['Status_Documento'].unique()
)

# Filtro de Tipo de Conta
tipo_conta = st.sidebar.multiselect(
    "Tipo de Conta:",
    options=df['Tipo_Conta'].unique(),
    default=df['Tipo_Conta'].unique()
)

# Aplicar filtros
df_filtrado = df[
    (df['Nivel_Risco'].isin(nivel_risco)) &
    (df['Tipo_Cliente'].isin(tipo_cliente)) &
    (df['Status_Documento'].isin(status_doc)) &
    (df['Tipo_Conta'].isin(tipo_conta))
]

if pep_filter != "Todos":
    df_filtrado = df_filtrado[df_filtrado['PEP_RCA'] == pep_filter]

# =============================================================================
# KPIs PRINCIPAIS
# =============================================================================
st.header("📊 Indicadores Principais (KPIs)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="👥 Total de Clientes",
        value=f"{len(df_filtrado):,}".replace(",", "."),
        delta=f"{(len(df_filtrado)/len(df)*100):.1f}% do total"
    )

with col2:
    clientes_alto_risco = len(df_filtrado[df_filtrado['Nivel_Risco'] == 'Alto'])
    st.metric(
        label="🔴 Risco ALTO",
        value=clientes_alto_risco,
        delta=f"{(clientes_alto_risco/len(df_filtrado)*100):.1f}%"
    )

with col3:
    peps = len(df_filtrado[df_filtrado['PEP_RCA'] == 'Sim'])
    st.metric(
        label="⚠️ PEPs (Pessoas Expostas)",
        value=peps,
        delta=f"{(peps/len(df_filtrado)*100):.1f}%"
    )

with col4:
    docs_expirados = len(df_filtrado[df_filtrado['Documentacao_Expirada'] == 'Sim'])
    st.metric(
        label="📄 Documentação Expirada",
        value=docs_expirados,
        delta=f"{(docs_expirados/len(df_filtrado)*100):.1f}%"
    )

with col5:
    nao_residentes = len(df_filtrado[df_filtrado['Cliente_Nao_Residente'] == 'Sim'])
    st.metric(
        label="🌍 Não Residentes",
        value=nao_residentes,
        delta=f"{(nao_residentes/len(df_filtrado)*100):.1f}%"
    )

st.markdown("---")

# =============================================================================
# TABS PRINCIPAIS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Perfil dos Clientes",
    "⚠️ Análise de Risco",
    "📋 Documentação & Compliance",
    "🔗 Análise para Rede Bayesiana",
    "🌍 Análise Geográfica",
    "📊 Tabela de Dados"
])

# =============================================================================
# TAB 1: PERFIL DOS CLIENTES
# =============================================================================
with tab1:
    st.header("📈 Perfil Demográfico dos Clientes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Sexo")
        sexo_counts = df_filtrado['Sexo'].value_counts().reset_index()
        sexo_counts.columns = ['Sexo', 'Contagem']
        fig_sexo = px.pie(
            sexo_counts, 
            values='Contagem', 
            names='Sexo',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_sexo.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sexo, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição por Faixa Etária")
        idade_counts = df_filtrado['Faixa_Etaria'].value_counts().reset_index()
        idade_counts.columns = ['Faixa_Etária', 'Contagem']
        idade_counts = idade_counts.sort_values('Faixa_Etária')
        fig_idade = px.bar(
            idade_counts, 
            x='Faixa_Etária', 
            y='Contagem',
            color='Faixa_Etária',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_idade.update_layout(showlegend=False)
        st.plotly_chart(fig_idade, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Distribuição por Tipo de Conta")
        conta_counts = df_filtrado['Tipo_Conta'].value_counts().reset_index()
        conta_counts.columns = ['Tipo de Conta', 'Contagem']
        fig_conta = px.bar(
            conta_counts, 
            x='Tipo de Conta', 
            y='Contagem',
            color='Tipo de Conta',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_conta.update_layout(showlegend=False)
        st.plotly_chart(fig_conta, use_container_width=True)
    
    with col4:
        st.subheader("Distribuição por Tipo de Cliente")
        tipo_counts = df_filtrado['Tipo_Cliente'].value_counts().reset_index()
        tipo_counts.columns = ['Tipo de Cliente', 'Contagem']
        fig_tipo = px.pie(
            tipo_counts, 
            values='Contagem', 
            names='Tipo de Cliente',
            color_discrete_sequence=px.colors.qualitative.Pastel1,
            hole=0.4
        )
        fig_tipo.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_tipo, use_container_width=True)
    
    st.subheader("📊 Tempo de Relacionamento com o Banco")
    tempo_counts = df_filtrado['Faixa_Tempo_Relacionamento'].value_counts().reset_index()
    tempo_counts.columns = ['Tempo de Relacionamento', 'Contagem']
    fig_tempo = px.area(
        tempo_counts, 
        x='Tempo de Relacionamento', 
        y='Contagem',
        color_discrete_sequence=['#0066cc']
    )
    fig_tempo.update_layout(showlegend=False)
    st.plotly_chart(fig_tempo, use_container_width=True)
    
    st.subheader("🏢 Top 15 Atividades Econômicas")
    atividade_counts = df_filtrado['Atividade_Economica'].value_counts().head(15).reset_index()
    atividade_counts.columns = ['Atividade Econômica', 'Contagem']
    fig_atividade = px.bar(
        atividade_counts, 
        y='Atividade Econômica', 
        x='Contagem',
        orientation='h',
        color='Contagem',
        color_continuous_scale='Blues'
    )
    fig_atividade.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_atividade, use_container_width=True)

# =============================================================================
# TAB 2: ANÁLISE DE RISCO
# =============================================================================
with tab2:
    st.header("⚠️ Análise Detalhada de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Distribuição Geral do Nível de Risco")
        risco_counts = df_filtrado['Nivel_Risco'].value_counts().reset_index()
        risco_counts.columns = ['Nível de Risco', 'Contagem']
        
        # Ordenar: Alto, Médio, Baixo
        ordem_risco = ['Alto', 'Médio', 'Baixo']
        risco_counts['ordem'] = risco_counts['Nível de Risco'].map(
            {r: i for i, r in enumerate(ordem_risco)}
        )
        risco_counts = risco_counts.sort_values('ordem')
        
        cores_risco = {'Alto': '#dc3545', 'Médio': '#ffc107', 'Baixo': '#28a745'}
        fig_risco = px.pie(
            risco_counts, 
            values='Contagem', 
            names='Nível de Risco',
            color='Nível de Risco',
            color_discrete_map=cores_risco,
            hole=0.5
        )
        fig_risco.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_risco, use_container_width=True)
    
    with col2:
        st.subheader("🔗 Risco vs PEP")
        risco_pep = pd.crosstab(df_filtrado['Nivel_Risco'], df_filtrado['PEP_RCA'])
        fig_risco_pep = px.bar(
            risco_pep.reset_index().melt(id_vars='Nivel_Risco'),
            x='Nivel_Risco',
            y='value',
            color='PEP_RCA',
            barmode='group',
            color_discrete_map={'Sim': '#dc3545', 'Não': '#28a745'},
            labels={'value': 'Contagem', 'Nivel_Risco': 'Nível de Risco', 'PEP_RCA': 'É PEP?'}
        )
        st.plotly_chart(fig_risco_pep, use_container_width=True)
    
    st.subheader("💼 Risco por Tipo de Conta")
    risco_conta = pd.crosstab(df_filtrado['Tipo_Conta'], df_filtrado['Nivel_Risco'])
    fig_risco_conta = px.bar(
        risco_conta.reset_index().melt(id_vars='Tipo_Conta'),
        x='Tipo_Conta',
        y='value',
        color='Nivel_Risco',
        barmode='stack',
        color_discrete_map={'Alto': '#dc3545', 'Médio': '#ffc107', 'Baixo': '#28a745'},
        labels={'value': 'Contagem', 'Nivel_Risco': 'Nível de Risco'}
    )
    st.plotly_chart(fig_risco_conta, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👤 Risco por Tipo de Cliente")
        risco_tipo = pd.crosstab(df_filtrado['Tipo_Cliente'], df_filtrado['Nivel_Risco'])
        fig_risco_tipo = px.bar(
            risco_tipo.reset_index().melt(id_vars='Tipo_Cliente'),
            x='Tipo_Cliente',
            y='value',
            color='Nivel_Risco',
            barmode='stack',
            color_discrete_map={'Alto': '#dc3545', 'Médio': '#ffc107', 'Baixo': '#28a745'},
            labels={'value': 'Contagem', 'Nivel_Risco': 'Nível de Risco'}
        )
        st.plotly_chart(fig_risco_tipo, use_container_width=True)
    
    with col4:
        st.subheader("🎂 Risco por Faixa Etária")
        risco_idade = pd.crosstab(df_filtrado['Faixa_Etaria'], df_filtrado['Nivel_Risco'])
        fig_risco_idade = px.bar(
            risco_idade.reset_index().melt(id_vars='Faixa_Etaria'),
            x='Faixa_Etaria',
            y='value',
            color='Nivel_Risco',
            barmode='stack',
            color_discrete_map={'Alto': '#dc3545', 'Médio': '#ffc107', 'Baixo': '#28a745'},
            labels={'value': 'Contagem', 'Nivel_Risco': 'Nível de Risco', 'Faixa_Etaria': 'Faixa Etária'}
        )
        st.plotly_chart(fig_risco_idade, use_container_width=True)
    
    st.subheader("🏢 Top 10 Atividades Econômicas com Maior Risco")
    # Calcular % de risco alto por atividade
    atividade_risco = df_filtrado.groupby('Atividade_Economica').agg(
        Total=('ID_Cliente', 'count'),
        Alto_Risco=('Nivel_Risco', lambda x: (x == 'Alto').sum())
    ).reset_index()
    atividade_risco['Pct_Alto_Risco'] = (atividade_risco['Alto_Risco'] / atividade_risco['Total'] * 100).round(2)
    atividade_risco = atividade_risco[atividade_risco['Total'] >= 5].sort_values('Pct_Alto_Risco', ascending=False).head(10)
    
    fig_atividade_risco = px.bar(
        atividade_risco,
        x='Pct_Alto_Risco',
        y='Atividade_Economica',
        orientation='h',
        color='Pct_Alto_Risco',
        color_continuous_scale='Reds',
        labels={'Pct_Alto_Risco': '% de Clientes em Risco Alto', 'Atividade_Economica': 'Atividade Econômica'}
    )
    st.plotly_chart(fig_atividade_risco, use_container_width=True)

# =============================================================================
# TAB 3: DOCUMENTAÇÃO & COMPLIANCE
# =============================================================================
with tab3:
    st.header("📋 Análise de Documentação e Compliance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Status dos Documentos")
        status_counts = df_filtrado['Status_Documento'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Contagem']
        cores_status = {
            'Válido': '#28a745', 
            'Expirado': '#dc3545', 
            'Quase Expirando': '#ffc107'
        }
        fig_status = px.pie(
            status_counts, 
            values='Contagem', 
            names='Status',
            color='Status',
            color_discrete_map=cores_status,
            hole=0.4
        )
        fig_status.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.subheader("🆔 Tipo de Documento Utilizado")
        tipo_doc = df_filtrado['Tipo_Documento'].value_counts().reset_index()
        tipo_doc.columns = ['Tipo de Documento', 'Contagem']
        fig_tipo_doc = px.bar(
            tipo_doc, 
            x='Tipo de Documento', 
            y='Contagem',
            color='Tipo de Documento',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_tipo_doc.update_layout(showlegend=False)
        st.plotly_chart(fig_tipo_doc, use_container_width=True)
    
    st.subheader("⚠️ Documentação Expirada vs Nível de Risco")
    doc_risco = pd.crosstab(df_filtrado['Documentacao_Expirada'], df_filtrado['Nivel_Risco'])
    fig_doc_risco = px.bar(
        doc_risco.reset_index().melt(id_vars='Documentacao_Expirada'),
        x='Documentacao_Expirada',
        y='value',
        color='Nivel_Risco',
        barmode='group',
        color_discrete_map={'Alto': '#dc3545', 'Médio': '#ffc107', 'Baixo': '#28a745'},
        labels={'value': 'Contagem', 'Documentacao_Expirada': 'Documentação Expirada?', 'Nivel_Risco': 'Nível de Risco'}
    )
    st.plotly_chart(fig_doc_risco, use_container_width=True)
    
    st.subheader("📅 Análise Temporal dos Documentos")
    df_tempo = df_filtrado.dropna(subset=['Data_Validade']).copy()
    df_tempo['Ano_Validade'] = df_tempo['Data_Validade'].dt.year
    
    # Filtrar anos válidos (remover outliers como 2099)
    df_tempo = df_tempo[(df_tempo['Ano_Validade'] >= 2020) & (df_tempo['Ano_Validade'] <= 2040)]
    
    ano_counts = df_tempo['Ano_Validade'].value_counts().reset_index()
    ano_counts.columns = ['Ano de Validade', 'Contagem']
    ano_counts = ano_counts.sort_values('Ano de Validade')
    
    fig_ano = px.line(
        ano_counts, 
        x='Ano de Validade', 
        y='Contagem',
        markers=True,
        color_discrete_sequence=['#0066cc']
    )
    fig_ano.add_vline(x=2026, line_dash="dash", line_color="red",
                     annotation_text="Ano Atual (2026)")
    st.plotly_chart(fig_ano, use_container_width=True)
    
    st.subheader("🔍 Status do Documento por Tipo de Cliente")
    status_tipo = pd.crosstab(df_filtrado['Tipo_Cliente'], df_filtrado['Status_Documento'])
    fig_status_tipo = px.bar(
        status_tipo.reset_index().melt(id_vars='Tipo_Cliente'),
        x='Tipo_Cliente',
        y='value',
        color='Status_Documento',
        barmode='stack',
        color_discrete_map={'Válido': '#28a745', 'Expirado': '#dc3545', 'Quase Expirando': '#ffc107'},
        labels={'value': 'Contagem', 'Status_Documento': 'Status do Documento'}
    )
    st.plotly_chart(fig_status_tipo, use_container_width=True)

# =============================================================================
# TAB 4: ANÁLISE PARA REDE BAYESIANA
# =============================================================================
with tab4:
    st.header("🔗 Análise de Dependência para Rede Bayesiana")
    st.info("""
    💡 **Importante para o TCC:** Esta seção apresenta as análises estatísticas fundamentais 
    para a construção da Rede Bayesiana. O **V de Cramer** mede a força de associação entre 
    variáveis categóricas (0 = sem associação, 1 = associação perfeita).
    """)
    
    st.subheader("📊 Matriz de Associação (V de Cramer)")
    
    # Selecionar variáveis para análise
    variaveis_rb = [
        'Nivel_Risco', 'PEP_RCA', 'Tipo_Conta', 'Tipo_Cliente', 
        'Status_Documento', 'Documentacao_Expirada', 'Sexo',
        'Cliente_Nao_Residente'
    ]
    
    with st.spinner("Calculando matriz de associação..."):
        matriz_assoc = calcular_matriz_associacao(df_filtrado, variaveis_rb)
    
    # Heatmap interativo
    fig_heatmap = px.imshow(
        matriz_assoc,
        text_auto=".2f",
        color_continuous_scale='RdYlBu_r',
        zmin=0, zmax=1,
        labels={'x': 'Variável', 'y': 'Variável', 'color': 'V de Cramer'},
        aspect="auto"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("""
    **📖 Interpretação do V de Cramer:**
    - **0.0 - 0.1**: Associação muito fraca ou inexistente
    - **0.1 - 0.3**: Associação fraca
    - **0.3 - 0.5**: Associação moderada
    - **0.5 - 1.0**: Associação forte
    """)
    
    st.subheader("🎯 Top 15 Pares de Variáveis com Maior Associação")
    
    # Extrair pares ordenados
    pares = []
    for i in range(len(variaveis_rb)):
        for j in range(i+1, len(variaveis_rb)):
            pares.append({
                'Variável 1': variaveis_rb[i],
                'Variável 2': variaveis_rb[j],
                'V de Cramer': matriz_assoc.iloc[i, j]
            })
    
    df_pares = pd.DataFrame(pares).sort_values('V de Cramer', ascending=False).head(15)
    
    fig_pares = px.bar(
        df_pares,
        x='V de Cramer',
        y=df_pares['Variável 1'] + ' ↔ ' + df_pares['Variável 2'],
        orientation='h',
        color='V de Cramer',
        color_continuous_scale='Viridis'
    )
    fig_pares.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig_pares, use_container_width=True)
    
    st.subheader("📈 Probabilidades Condicionais (Tabelas para Rede Bayesiana)")
    
    st.markdown("""
    **P(A|B) - Probabilidade Condicional:** A probabilidade de A ocorrer dado que B ocorreu.
    Estas tabelas são a base para definir as **Tabelas de Probabilidade Condicional (CPT)** 
    da Rede Bayesiana.
    """)
    
    # Seleção de variáveis para análise condicional
    col1, col2 = st.columns(2)
    with col1:
        var_pai = st.selectbox(
            "Variável PAI (condicionante):",
            options=variaveis_rb,
            index=variaveis_rb.index('Nivel_Risco')
        )
    with col2:
        var_filha = st.selectbox(
            "Variável FILHA (condicionada):",
            options=[v for v in variaveis_rb if v != var_pai],
            index=0
        )
    
    # Calcular probabilidade condicional
    crosstab = pd.crosstab(df_filtrado[var_pai], df_filtrado[var_filha], normalize='index') * 100
    
    st.markdown(f"**P({var_filha} | {var_pai})** - Probabilidade em %")
    
    fig_cond = px.imshow(
        crosstab,
        text_auto=".1f",
        color_continuous_scale='Blues',
        labels={'x': var_filha, 'y': var_pai, 'color': 'Probabilidade (%)'},
        aspect="auto"
    )
    st.plotly_chart(fig_cond, use_container_width=True)
    
    # Mostrar tabela numérica
    with st.expander("📋 Ver Tabela Numérica de Probabilidades Condicionais"):
        st.dataframe(crosstab.round(2))
    
    st.subheader("🌐 Estrutura Sugerida para a Rede Bayesiana")
    
    st.markdown("""
    Com base nas associações calculadas, sugere-se a seguinte estrutura para a Rede Bayesiana:
    
    **Nó Central (Target):** `Nivel_Risco`
    
    **Nós Pais (Evidências):**
    - `PEP_RCA` → forte influência no risco
    - `Tipo_Conta` → indica perfil financeiro
    - `Cliente_Nao_Residente` → risco jurisdicional
    - `Documentacao_Expirada` → indicador de compliance
    - `Tipo_Cliente` → perfil do cliente
    """)
    
    # Diagrama simplificado da rede
    st.markdown("""
    ```
                    ┌─────────────┐
                    │   PEP_RCA   │
                    └──────┬──────┘
                           │
    ┌─────────────┐        │        ┌─────────────────┐
    │ Tipo_Conta  │────────┼────────│ Cliente_Nao_Res.│
    └──────┬──────┘        │        └────────┬────────┘
           │               │                 │
           └───────────────┼─────────────────┘
                           ▼
                    ┌─────────────────┐
                    │   Nivel_Risco   │  ← NÓ CENTRAL
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Doc.Expirada │ │ Tipo_Cliente │ │ Atividade    │
    └──────────────┘ └──────────────┘ └──────────────┘
    ```
    """)

# =============================================================================
# TAB 5: ANÁLISE GEOGRÁFICA
# =============================================================================
with tab5:
    st.header("🌍 Análise Geográfica dos Clientes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏳️ Top 15 Nacionalidades")
        nac_counts = df_filtrado['Nacionalidade_Primaria'].value_counts().head(15).reset_index()
        nac_counts.columns = ['Nacionalidade', 'Contagem']
        fig_nac = px.bar(
            nac_counts, 
            y='Nacionalidade', 
            x='Contagem',
            orientation='h',
            color='Contagem',
            color_continuous_scale='Tealgrn'
        )
        fig_nac.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_nac, use_container_width=True)
    
    with col2:
        st.subheader("🌐 Top 15 Países de Residência")
        pais_counts = df_filtrado['Pais_Residencia'].value_counts().head(15).reset_index()
        pais_counts.columns = ['País', 'Contagem']
        fig_pais = px.bar(
            pais_counts, 
            y='País', 
            x='Contagem',
            orientation='h',
            color='Contagem',
            color_continuous_scale='PuBu'
        )
        fig_pais.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pais, use_container_width=True)
    
    st.subheader("🗺️ Risco por País de Residência (Top 15)")
    pais_risco = df_filtrado.groupby(['Pais_Residencia', 'Nivel_Risco']).size().reset_index(name='Contagem')
    top_paises = df_filtrado['Pais_Residencia'].value_counts().head(15).index
    pais_risco = pais_risco[pais_risco['Pais_Residencia'].isin(top_paises)]
    
    fig_pais_risco = px.bar(
        pais_risco,
        x='Pais_Residencia',
        y='Contagem',
        color='Nivel_Risco',
        barmode='stack',
        color_discrete_map={'Alto': '#dc3545', 'Médio': '#ffc107', 'Baixo': '#28a745'},
        labels={'Pais_Residencia': 'País de Residência', 'Contagem': 'Total de Clientes'}
    )
    fig_pais_risco.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pais_risco, use_container_width=True)
    
    st.subheader("🎯 % de Clientes em Risco ALTO por País (min. 10 clientes)")
    pais_risco_pct = df_filtrado.groupby('Pais_Residencia').agg(
        Total=('ID_Cliente', 'count'),
        Alto_Risco=('Nivel_Risco', lambda x: (x == 'Alto').sum())
    ).reset_index()
    pais_risco_pct['Pct_Alto'] = (pais_risco_pct['Alto_Risco'] / pais_risco_pct['Total'] * 100).round(2)
    pais_risco_pct = pais_risco_pct[pais_risco_pct['Total'] >= 10].sort_values('Pct_Alto', ascending=False).head(15)
    
    fig_pais_pct = px.bar(
        pais_risco_pct,
        x='Pct_Alto',
        y='Pais_Residencia',
        orientation='h',
        color='Pct_Alto',
        color_continuous_scale='Reds',
        labels={'Pct_Alto': '% em Risco Alto', 'Pais_Residencia': 'País'}
    )
    fig_pais_pct.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_pais_pct, use_container_width=True)

# =============================================================================
# TAB 6: TABELA DE DADOS
# =============================================================================
with tab6:
    st.header("📊 Base de Dados Completa")
    
    st.info(f"📌 Exibindo {len(df_filtrado)} registros filtrados de um total de {len(df)} clientes.")
    
    # Seleção de colunas
    colunas_disponiveis = df_filtrado.columns.tolist()
    colunas_selecionadas = st.multiselect(
        "Selecione as colunas para visualizar:",
        options=colunas_disponiveis,
        default=['ID_Cliente', 'Tipo_Cliente', 'Nacionalidade_Primaria', 'Nivel_Risco', 
                 'PEP_RCA', 'Tipo_Conta', 'Status_Documento', 'Idade', 'Sexo']
    )
    
    if colunas_selecionadas:
        # Filtro de busca
        busca = st.text_input("🔍 Buscar na tabela (qualquer coluna):")
        
        df_exibir = df_filtrado[colunas_selecionadas]
        
        if busca:
            mask = df_exibir.astype(str).apply(lambda x: x.str.contains(busca, case=False)).any(axis=1)
            df_exibir = df_exibir[mask]
        
        st.dataframe(
            df_exibir,
            use_container_width=True,
            height=500
        )
        
        # Botão de download
        csv = df_exibir.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar dados filtrados em CSV",
            data=csv,
            file_name="kyc_dados_filtrados.csv",
            mime="text/csv"
        )
    
    st.subheader("📈 Estatísticas Descritivas")
    st.dataframe(df_filtrado.describe().round(2), use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🎓 Trabalho de Conclusão de Curso</strong></p>
    <p>Implementação de um Sistema de Avaliação de Risco KYC com Redes Bayesianas no Contexto de Cabo Verde</p>
    <p><em>Dashboard desenvolvido para análise e visualização de dados - 2026</em></p>
</div>
""", unsafe_allow_html=True)