"""
Dashboard de Gestão de Risco Operacional - KYC & Modelagem Bayesiana
TFC Matemática Aplicada - 2026
Autor: Manilson Semedo
"""

# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÃO INICIAL
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import warnings; warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES CENTRALIZADAS
# =============================================================================
st.set_page_config(page_title="Dashboard Risco Operacional - KYC", page_icon="🏦", layout="wide")

CONFIG = {
    'RISCO': {'ordem': ['Baixo', 'Médio', 'Alto'], 'cores': {'Baixo': '#2ECC71', 'Médio': '#F39C12', 'Alto': '#E74C3C'}, 'limiares': {'Baixo': 2, 'Medio': 5}},
    'ILHAS': {'nomes': ['Santiago', 'Sal', 'São Vicente', 'Boa Vista', 'Maio', 'Fogo', 'Brava', 'Santo Antão', 'São Nicolau'], 
              'probs': np.array([0.55, 0.10, 0.20, 0.06, 0.015, 0.04, 0.005, 0.03, 0.005])},
    'NACIONALIDADES': ['Cabo-Verdiana', 'Portuguesa', 'Brasileira', 'Angolana', 'Guineense', 'Senegalesa', 'Nigeriana', 'Outra'],
    'ATIVIDADES': ['Comércio por grosso e a retalho', 'Atividades de construção', 'Alojamento e restauração', 'Atividades financeiras e de seguros', 'Atividades imobiliárias', 'Atividades profissionais, científicas e técnicas', 'Atividades administrativas e de apoio', 'Administração pública e defesa', 'Educação', 'Saúde humana e apoio social', 'Artes, entretenimento e recreação', 'Outras atividades de serviços', 'Agricultura, pecuária e pesca', 'Indústria transformadora', 'Transporte e armazenagem', 'Informação e comunicação', 'Forças armadas', 'Estudante', 'Desempregado', 'Reformado', 'Outra'],
    'PAISES': ['Portugal', 'Estados Unidos', 'Brasil', 'Angola', 'Itália', 'França', 'Holanda', 'Luxemburgo', 'Suíça', 'Espanha', 'Alemanha', 'Reino Unido', 'Canadá', 'Senegal', 'Guiné-Bissau', 'Nigéria', 'China', 'Outro']
}
CONFIG['ILHAS']['probs'] /= CONFIG['ILHAS']['probs'].sum()
PROB_ATIV = np.array([0.15, 0.12, 0.10, 0.08, 0.08, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
PROB_ATIV /= PROB_ATIV.sum()

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================
def classificar_risco(score):
    """Converte score numérico em categoria de risco (Baixo ≤2, Médio ≤5, Alto >5)"""
    return 'Baixo' if score <= CONFIG['RISCO']['limiares']['Baixo'] else 'Médio' if score <= CONFIG['RISCO']['limiares']['Medio'] else 'Alto'

def calcular_score(tipo, incidente, reclamacao, rend, tempo, pep, doc_status=None):
    """Calcula pontuação de risco baseada em múltiplos fatores"""
    score = {'Não Residente': 2, 'Emigrante': 1, 'Residente': 0}.get(tipo, 0)
    score += 2 if pep == 'Sim' else 0
    score += {'3+ incidentes': 2, '1-2 incidentes': 1, 'Nenhum': 0}.get(incidente, 0)
    score += 2 if reclamacao >= 3 else 1 if reclamacao >= 1 else 0
    score += (1 if rend < 15000 else 0) + (1 if tempo < 1 else 0)
    if doc_status:
        score += 2 if doc_status == 'Expirado' else 1 if doc_status == 'Quase Expirando' else 0
    return score

def criar_grafico_risco_barras(df, x_col, titulo, stacked=False):
    """Cria gráfico de barras percentuais com cores padronizadas por nível de risco"""
    agg = df.groupby([x_col, 'Nivel_Risco']).size().unstack(fill_value=0).reindex(columns=CONFIG['RISCO']['ordem']).fillna(0)
    agg_pct = agg.div(agg.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for risco in CONFIG['RISCO']['ordem']:
        if risco in agg_pct.columns:
            fig.add_trace(go.Bar(name=risco, x=agg_pct.index, y=agg_pct[risco], marker_color=CONFIG['RISCO']['cores'][risco]))
    fig.update_layout(title=titulo, barmode='stack' if stacked else 'group', xaxis_title=x_col.replace('_', ' '), yaxis_title='Percentual (%)')
    return fig

def criar_pizza_risco(df, col, titulo, hole=0.4):
    """Cria gráfico de rosca com cores padronizadas"""
    return px.pie(df, names=col, title=titulo, hole=hole, color=col, color_discrete_map=CONFIG['RISCO']['cores'], category_orders={col: CONFIG['RISCO']['ordem']}).update_traces(textinfo='percent+label')

# =============================================================================
# GERAÇÃO DE DADOS SINTÉTICOS (MODIFICADO)
# =============================================================================
@st.cache_data
def gerar_base_clientes(n_clientes=5000, seed=42):
    """Gera base de dados sintética representativa do mercado bancário cabo-verdiano"""
    np.random.seed(seed)
    data_atual = datetime(2024, 12, 1)
    
    # IDs e geografia
    ids = [f"CLI_{str(i).zfill(6)}" for i in range(1, n_clientes + 1)]
    ilhas = np.random.choice(CONFIG['ILHAS']['nomes'], n_clientes, p=CONFIG['ILHAS']['probs'])
    idades = np.clip(np.random.normal(40, 15, n_clientes).astype(int), 18, 90)
    sexos = np.random.choice(['Masculino', 'Feminino'], n_clientes, p=[0.48, 0.52])
    nacs = np.random.choice(CONFIG['NACIONALIDADES'], n_clientes, p=[0.78, 0.08, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01])
    
    # Múltipla nacionalidade
    multi_nac = [np.random.choice(['Cabo-Verdiana', 'Cabo-Verdiana/Portuguesa', 'Cabo-Verdiana/Outra'], p=[0.85, 0.12, 0.03]) if n == 'Cabo-Verdiana' else 
                 np.random.choice(['Portuguesa', 'Portuguesa/Cabo-Verdiana', 'Portuguesa/Outra'], p=[0.80, 0.15, 0.05]) if n == 'Portuguesa' else n for n in nacs]
    
    # Tipo de cliente e residência (AJUSTADO: aumentar residentes, reduzir não residentes)
    tipos, paises, cond_res = [], [], []
    for n in nacs:
        # Aumentar proporção de residentes e reduzir não residentes
        if n == 'Cabo-Verdiana': 
            t = np.random.choice(['Residente', 'Emigrante', 'Não Residente'], p=[0.70, 0.20, 0.10])  # Antes: [0.60, 0.25, 0.15]
        else: 
            t = np.random.choice(['Residente', 'Não Residente'], p=[0.80, 0.20])  # Antes: [0.70, 0.30]
        tipos.append(t)
        if t == 'Residente':
            paises.append('Cabo Verde'); cond_res.append('Sim')
        elif t == 'Emigrante':
            paises.append(np.random.choice(['Portugal', 'Estados Unidos', 'Brasil', 'Angola', 'Itália', 'França', 'Holanda', 'Luxemburgo', 'Suíça', 'Espanha'], p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]))
            cond_res.append('Não')
        else:
            paises.append(np.random.choice(['Portugal', 'Estados Unidos', 'Brasil', 'Angola', 'Itália'] if n == 'Cabo-Verdiana' else ['Portugal', 'Brasil', 'Angola', 'Nigéria', 'China', 'Senegal', 'Outro'], p=[0.40, 0.25, 0.15, 0.12, 0.08] if n == 'Cabo-Verdiana' else [0.20, 0.20, 0.15, 0.15, 0.15, 0.10, 0.05]))
            cond_res.append('Não')
    
    # Financeiro - AJUSTAR: aumentar rendimentos para reduzir risco
    rendimentos = np.round(np.random.lognormal(10.2, 0.8, n_clientes), 0).astype(int)  # Antes: (9.8, 0.9) - rendimentos mais altos = menos risco
    
    tipos_conta = [np.random.choice(['Conta Básica', 'Conta Standard', 'Conta Premium'], p=[0.40, 0.45, 0.15]) if r < 20000 else
                   np.random.choice(['Conta Básica', 'Conta Standard', 'Conta Premium'], p=[0.15, 0.55, 0.30]) if r < 50000 else
                   np.random.choice(['Conta Standard', 'Conta Premium', 'Conta Private'], p=[0.10, 0.50, 0.40]) for r in rendimentos]  # Mais contas premium = menos risco
    tempos = np.clip(np.random.exponential(8, n_clientes).astype(int), 0, 40)  # Antes: 7 - mais tempo de relacionamento = menos risco
    n_produtos = [min(min(tempo // 3, 5) + (np.random.randint(1, 4) if conta in ['Conta Premium', 'Conta Private'] else np.random.randint(0, 2)), 8) for tempo, conta in zip(tempos, tipos_conta)]
    
    # Atividade e PEP - REDUZIR PEP
    atividades = np.random.choice(CONFIG['ATIVIDADES'], n_clientes, p=PROB_ATIV)
    peps = ['Sim' if np.random.random() < (0.05 if a in ['Administração pública e defesa', 'Forças armadas', 'Atividades financeiras e de seguros'] else 0.02 if a in ['Atividades profissionais, científicas e técnicas', 'Atividades imobiliárias'] else 0.005) else 'Não' for a in atividades]  # Reduzir probabilidade de PEP
    
    # Histórico de incidentes e reclamações - REDUZIR incidentes
    incidentes = []
    for t, p in zip(tipos, peps):
        # Reduzir significativamente a probabilidade de incidentes
        prob = min(0.03 + (0.05 if t in ['Emigrante', 'Não Residente'] else 0) + (0.02 if p == 'Sim' else 0), 0.25)  # Antes: 0.05 base, agora 0.03
        incidentes.append(np.random.choice(['Nenhum', '1-2 incidentes', '3+ incidentes'], p=[1-prob*2, prob*1.5, prob*0.5]))
    
    reclamacoes = []
    for inc, t in zip(incidentes, tipos):
        base = 0 if t in ['Emigrante', 'Não Residente'] else 0  # Antes: base = 1, agora 0
        if inc == 'Nenhum':
            probs = [0.90, 0.08, 0.02]  # Antes: [0.85, 0.12, 0.03]
        elif inc == '1-2 incidentes':
            probs = [0.50, 0.35, 0.15]  # Antes: [0.40, 0.35, 0.25]
        else:
            probs = [0.30, 0.35, 0.35]  # Antes: [0.20, 0.30, 0.50]
        probs = np.array(probs) / sum(probs)
        reclamacoes.append(np.random.choice([0, 1, 2], p=probs) + base)
    
    # Classificação inicial
    riscos = [classificar_risco(calcular_score(t, i, r, rend, temp, p)) for t, i, r, rend, temp, p in zip(tipos, incidentes, reclamacoes, rendimentos, tempos, peps)]
    
    # Datas KYC - AJUSTAR para mais risco baixo
    datas_ult, datas_prox, status_atual = [], [], []
    for risco, tempo in zip(riscos, tempos):
        ultima = data_atual - timedelta(days=np.random.randint(30, max(int(tempo * 365), 31)))
        datas_ult.append(ultima)
        dias_add = 5*365 if risco == 'Baixo' else 2*365 if risco == 'Médio' else 365
        proxima = ultima + timedelta(days=dias_add)
        datas_prox.append(proxima)
        status_atual.append('Atrasado' if (proxima - data_atual).days < 0 else 'A vencer' if (proxima - data_atual).days <= 90 else 'Em dia')
    
    # Documentação - AJUSTAR para reduzir documentação expirada
    tipos_doc, datas_emiss, datas_val, status_doc = [], [], [], []
    for tipo_cli, nac, idade in zip(tipos, nacs, idades):
        if tipo_cli == 'Emigrante':
            doc = np.random.choice(['Passaporte CV', 'Prova de Emigrante', 'BI Cabo-Verdiano'], p=[0.55, 0.30, 0.15])  # Antes: [0.50, 0.35, 0.15]
        elif tipo_cli == 'Não Residente':
            doc = np.random.choice(['Passaporte CV', 'BI Cabo-Verdiano', 'CC'], p=[0.65, 0.25, 0.10]) if nac == 'Cabo-Verdiana' else np.random.choice(['Passaporte Estrangeiro', 'Autorização Residência CV', 'BI Estrangeiro'], p=[0.75, 0.15, 0.10])
        else:
            if idade < 18:
                doc = np.random.choice(['Certidão Nascimento', 'BI Cabo-Verdiano (Menor)'], p=[0.85, 0.15])
            elif nac == 'Cabo-Verdiana':
                doc = np.random.choice(['BI Cabo-Verdiano', 'CC', 'Passaporte CV'], p=[0.80, 0.12, 0.08])
            else:
                doc = np.random.choice(['Passaporte Estrangeiro', 'Autorização Residência CV', 'BI Estrangeiro'], p=[0.55, 0.25, 0.20])
        tipos_doc.append(doc)
        emissao = data_atual - timedelta(days=np.random.randint(30, 3650))
        datas_emiss.append(emissao)
        anos = 5 if doc in ['BI Cabo-Verdiano', 'BI Cabo-Verdiano (Menor)', 'CC'] or 'Passaporte' in doc else 15 if doc == 'Certidão Nascimento' else 2 if doc == 'Prova de Emigrante' else 1 if doc == 'Autorização Residência CV' else 5
        validade = emissao + timedelta(days=anos*365)
        datas_val.append(validade)
        dias_exp = (validade - data_atual).days
        # Reduzir documentação expirada
        if dias_exp < 0:
            status_doc.append('Expirado' if np.random.random() < 0.7 else 'Quase Expirando')  # Antes: sempre 'Expirado'
        else:
            status_doc.append('Expirado' if dias_exp < 0 else 'Quase Expirando' if dias_exp <= 60 else 'Válido')  # Antes: 90 dias
    
    # Reclassificação final e scores
    riscos_finais = [classificar_risco(calcular_score(t, i, r, rend, temp, p, d)) for t, i, r, rend, temp, p, d in zip(tipos, incidentes, reclamacoes, rendimentos, tempos, peps, status_doc)]
    
    # Ajustar manualmente a distribuição final para garantir mais risco baixo
    # Converter alguns clientes de risco médio para baixo
    riscos_finais_arr = np.array(riscos_finais)
    indices_medio = np.where(riscos_finais_arr == 'Médio')[0]
    n_converter = int(len(indices_medio) * 0.3)  # Converter 30% dos médios para baixo
    if n_converter > 0:
        indices_converter = np.random.choice(indices_medio, n_converter, replace=False)
        for idx in indices_converter:
            riscos_finais[idx] = 'Baixo'
    
    scores_conf = [min((25 if d == 'Válido' else 15 if d == 'Quase Expirando' else 0) + (25 if s == 'Em dia' else 15 if s == 'A vencer' else 0) + (20 if i == 'Nenhum' else 10 if i == '1-2 incidentes' else 0) + (15 if r == 0 else 8 if r <= 2 else 0) + (15 if t == 'Residente' else 10 if t == 'Emigrante' else 0) + (10 if p == 'Não' else 0), 100) for d, s, i, r, t, p in zip(status_doc, status_atual, incidentes, reclamacoes, tipos, peps)]
    freq_atual = ['5 anos' if r == 'Baixo' else '2 anos' if r == 'Médio' else '1 ano' for r in riscos_finais]
    
    # DataFrame final
    df = pd.DataFrame({
        'ID_Cliente': ids, 'Ilha_Balcao': ilhas, 'Idade': idades, 'Sexo': sexos,
        'Nacionalidade_Primaria': nacs, 'Multipla_Nacionalidade': multi_nac,
        'Tipo_Cliente': tipos, 'Pais_Residencia': paises, 'Condicao_Residente': cond_res,
        'Rendimento_Mensal': rendimentos, 'Tipo_Conta': tipos_conta,
        'Tempo_Relacionamento': tempos, 'Numero_Produtos': n_produtos,
        'Atividade_Economica': atividades, 'PEP_RCA': peps,
        'Historico_Incidentes': incidentes, 'Numero_Reclamacoes': reclamacoes,
        'Nivel_Risco': riscos_finais, 'Data_Ultima_Atualizacao': datas_ult,
        'Proxima_Data_Atualizacao': datas_prox, 'Status_Atualizacao': status_atual,
        'Tipo_Documento': tipos_doc, 'Data_Emissao': datas_emiss,
        'Data_Validade': datas_val, 'Status_Documento': status_doc,
        'Score_Conformidade': scores_conf, 'Frequencia_Atualizacao': freq_atual
    })
    
    # Variáveis binárias para a Rede Bayesiana
    df['Documentacao_Expirada'] = df['Status_Documento'].apply(lambda x: 'Sim' if x == 'Expirado' else 'Não')
    df['Atraso_Atualizacao'] = df['Status_Atualizacao'].apply(lambda x: 'Sim' if x == 'Atrasado' else 'Não')
    df['Cliente_Nao_Residente'] = df['Tipo_Cliente'].apply(lambda x: 'Sim' if x == 'Não Residente' else 'Não')
    return df

# =============================================================================
# REDE BAYESIANA
# =============================================================================
@st.cache_resource
def construir_rede_bayesiana():
    """Constrói a Rede Bayesiana com estrutura e probabilidades condicionais"""
    modelo = DiscreteBayesianNetwork([
        ('Documentacao_Expirada', 'Nivel_Risco'), ('Atraso_Atualizacao', 'Nivel_Risco'),
        ('Cliente_Nao_Residente', 'Nivel_Risco'), ('PEP_RCA', 'Nivel_Risco'),
        ('Historico_Incidentes', 'Nivel_Risco'), ('Numero_Reclamacoes', 'Nivel_Risco'),
        ('Score_Conformidade', 'Nivel_Risco'), ('Frequencia_Atualizacao', 'Score_Conformidade')
    ])
    
    cpds = [
        TabularCPD('Documentacao_Expirada', 2, [[0.85], [0.15]], state_names={'Documentacao_Expirada': ['Não', 'Sim']}),
        TabularCPD('Atraso_Atualizacao', 2, [[0.88], [0.12]], state_names={'Atraso_Atualizacao': ['Não', 'Sim']}),
        TabularCPD('Cliente_Nao_Residente', 2, [[0.75], [0.25]], state_names={'Cliente_Nao_Residente': ['Não', 'Sim']}),
        TabularCPD('PEP_RCA', 2, [[0.97], [0.03]], state_names={'PEP_RCA': ['Não', 'Sim']}),
        TabularCPD('Historico_Incidentes', 3, [[0.75], [0.20], [0.05]], state_names={'Historico_Incidentes': ['Nenhum', '1-2 incidentes', '3+ incidentes']}),
        TabularCPD('Numero_Reclamacoes', 3, [[0.60], [0.30], [0.10]], state_names={'Numero_Reclamacoes': ['Baixo', 'Médio', 'Alto']}),
        TabularCPD('Frequencia_Atualizacao', 3, [[0.50], [0.35], [0.15]], state_names={'Frequencia_Atualizacao': ['5 anos', '2 anos', '1 ano']}),
        TabularCPD('Score_Conformidade', 3, [[0.70, 0.40, 0.20], [0.25, 0.45, 0.40], [0.05, 0.15, 0.40]], evidence=['Frequencia_Atualizacao'], evidence_card=[3], state_names={'Score_Conformidade': ['Alto', 'Médio', 'Baixo'], 'Frequencia_Atualizacao': ['5 anos', '2 anos', '1 ano']})
    ]
    
    # Gera CPD do Nível de Risco (432 combinações) - CORRIGIDO
    valores = []
    for d in [0, 1]:      # documentacao
        for a in [0, 1]:  # atraso
            for nr in [0, 1]:  # nao_residente
                for p in [0, 1]:  # pep
                    for i in [0, 1, 2]:  # incidentes
                        for r in [0, 1, 2]:  # reclamacoes
                            for sc in [0, 1, 2]:  # score
                                score_calc = d*2 + a*2 + nr*3 + p*3 + (i>0)*1 + (i>1)*2 + (r>0)*1 + (r>1)*2 + (sc>0)*1 + (sc>1)*2
                                if score_calc <= 3:
                                    p_vals = [0.75, 0.20, 0.05]
                                elif score_calc <= 7:
                                    p_vals = [0.40, 0.45, 0.15]
                                elif score_calc <= 12:
                                    p_vals = [0.20, 0.50, 0.30]
                                else:
                                    p_vals = [0.05, 0.25, 0.70]
                                valores.append(p_vals)
    
    cpds.append(TabularCPD('Nivel_Risco', 3, np.array(valores).T.tolist(),
        evidence=['Documentacao_Expirada', 'Atraso_Atualizacao', 'Cliente_Nao_Residente', 'PEP_RCA', 'Historico_Incidentes', 'Numero_Reclamacoes', 'Score_Conformidade'],
        evidence_card=[2, 2, 2, 2, 3, 3, 3],
        state_names={'Nivel_Risco': ['Baixo', 'Médio', 'Alto'], 'Documentacao_Expirada': ['Não', 'Sim'], 'Atraso_Atualizacao': ['Não', 'Sim'], 'Cliente_Nao_Residente': ['Não', 'Sim'], 'PEP_RCA': ['Não', 'Sim'], 'Historico_Incidentes': ['Nenhum', '1-2 incidentes', '3+ incidentes'], 'Numero_Reclamacoes': ['Baixo', 'Médio', 'Alto'], 'Score_Conformidade': ['Alto', 'Médio', 'Baixo']}))
    
    modelo.add_cpds(*cpds)
    return (modelo, VariableElimination(modelo)) if modelo.check_model() else (None, None)

def calcular_probabilidade_risco(inferencia, evidencias):
    """Executa inferência probabilística P(Risco | Evidências)"""
    try:
        resultado = inferencia.query(variables=['Nivel_Risco'], evidence={k:v for k,v in evidencias.items() if v != 'Não informado'})
        return {estado: resultado.values[i] for i, estado in enumerate(resultado.state_names['Nivel_Risco'])}
    except Exception as e:
        st.error(f"Erro na inferência: {str(e)}")
        return None

def criar_grafo_rede(modelo):
    """Visualiza a estrutura da Rede Bayesiana como grafo direcionado"""
    G = nx.DiGraph(modelo.edges())
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0, x1, y1 = *pos[u], *pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    
    node_x, node_y, node_text, node_color = [], [], [], []
    cores_no = {'Nivel_Risco': '#FF6B6B', 'Score_Conformidade': '#4ECDC4', 'Cliente_Nao_Residente': '#FFA07A', 'PEP_RCA': '#FFA07A'}
    for node in G.nodes():
        node_x.append(pos[node][0]); node_y.append(pos[node][1])
        node_text.append(node.replace('_', ' '))
        node_color.append(cores_no.get(node, '#95E1D3'))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), mode='lines'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", marker=dict(size=40, color=node_color, line=dict(width=2, color='DarkSlateGrey'))))
    fig.update_layout(title='Estrutura da Rede Bayesiana', showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500)
    return fig

def calcular_score_manual(tipo, rend, incidente, doc_exp, freq, tempo, prod, reclam, pais=None, ativ=None, pep='Não', multi_nac=None):
    """Matriz de Risco: cálculo determinístico com 12 fatores para avaliação manual"""
    score = 0
    fatores = []
    
    pesos_tipo = {'Não Residente': (4, "Cliente Não Residente (+4)"), 'Emigrante': (3, "Cliente Emigrante (+3)"), 'Residente': (0, None)}
    s, f = pesos_tipo[tipo]; score += s; f and fatores.append(f)
    
    if pep == 'Sim': score += 4; fatores.append("PEP/RCA (+4)")
    
    if rend < 15000: score += 3; fatores.append("Rendimento muito baixo (+3)")
    elif rend < 30000: score += 2; fatores.append("Rendimento baixo (+2)")
    elif rend < 50000: score += 1
    
    pesos_inc = {'3+ incidentes': (4, "Histórico crítico de incidentes (+4)"), '1-2 incidentes': (2, "Histórico de incidentes (+2)"), 'Nenhum': (0, None)}
    s, f = pesos_inc[incidente]; score += s; f and fatores.append(f)
    
    if doc_exp == 'Sim': score += 3; fatores.append("Documentação expirada (+3)")
    score += 2 if freq == '1 ano' else 1 if freq == '2 anos' else 0
    score += 2 if tempo < 1 else 1 if tempo < 3 else 0
    if prod < 2: score += 1; fatores.append("Poucos produtos (+1)")
    
    pesos_rec = {'Alto (3+)': (3, "Múltiplas reclamações (+3)"), 'Médio (1-2)': (2, "Reclamações registradas (+2)"), 'Baixo (0)': (0, None)}
    s, f = pesos_rec[reclam]; score += s; f and fatores.append(f)
    
    paises_alto, paises_medio = ['Nigéria', 'China', 'Rússia', 'Irã', 'Coreia do Norte'], ['Angola', 'Guiné-Bissau', 'Senegal', 'Brasil']
    if pais:
        if any(p in pais for p in paises_alto): score += 2; fatores.append(f"País de alto risco: {pais} (+2)")
        elif any(p in pais for p in paises_medio): score += 1; fatores.append(f"País de atenção: {pais} (+1)")
    
    ativ_alto, ativ_medio = ['Atividades financeiras e de seguros', 'Atividades imobiliárias', 'Administração pública e defesa', 'Informação e comunicação'], ['Comércio por grosso e a retalho', 'Atividades de construção', 'Alojamento e restauração', 'Outras atividades de serviços']
    if ativ:
        if any(a in ativ for a in ativ_alto): score += 2; fatores.append(f"Atividade de alto risco: {ativ} (+2)")
        elif any(a in ativ for a in ativ_medio): score += 1
    
    if multi_nac and '/' in str(multi_nac): score += 1; fatores.append("Múltipla nacionalidade (+1)")
    
    classif = 'Baixo' if score <= 6 else 'Médio' if score <= 14 else 'Alto'
    return score, classif, min(score/25, 0.95), fatores

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================
def main():
    st.title("🏦 Dashboard de Gestão de Risco Operacional - KYC")
    st.markdown("**TFC Matemática Aplicada** | *Business Intelligence e Modelagem Probabilística*")
    st.markdown("**Banco Comercial de Cabo Verde - Moeda: CVE**")
    st.divider()
    
    with st.sidebar:
        st.header("⚙️ Configurações")
        n_clientes = st.slider("Número de Clientes", 1000, 10000, 5000, 1000)
        if st.button("🔄 Regenerar Dados", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        with st.expander("📚 Metodologia"):
            st.markdown("- **Rede Bayesiana** (pgmpy): Inferência probabilística\n- **Framework KYC**: Classificação determinística\n- **Contexto CV**: 9 ilhas, 3 tipos de clientes")
        st.caption("© 2024 - TFC Matemática Aplicada")
    
    with st.spinner("Gerando dados..."): df = gerar_base_clientes(n_clientes)
    with st.spinner("Construindo modelo..."): modelo, inferencia = construir_rede_bayesiana()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visão Executiva", "🔄 Monitorização KYC", "📋 Gestão Documental", "🧠 Rede Bayesiana", "⚖️ Matriz de Risco"])
    
    # ========== ABA 1: VISÃO EXECUTIVA ==========
    with tab1:
        st.header("Visão Executiva")
        cols = st.columns(5)
        metricas = [
            ("Total de Clientes", f"{len(df):,}", None),
            ("% Risco Alto", f"{(df['Nivel_Risco'] == 'Alto').mean()*100:.1f}%", "Clientes classificados como risco alto"),
            ("% Docs Expirados", f"{(df['Status_Documento'] == 'Expirado').mean()*100:.1f}%", "Documentos fora da validade"),
            ("% Não Residentes", f"{(df['Tipo_Cliente'] == 'Não Residente').mean()*100:.1f}%", "Clientes não residentes"),
            ("Score Médio", f"{df['Score_Conformidade'].mean():.1f}/100", "Média do score de conformidade")
        ]
        for col, (t, v, h) in zip(cols, metricas):
            with col: st.metric(t, v, help=h)
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(criar_pizza_risco(df, 'Nivel_Risco', 'Distribuição de Risco'), use_container_width=True)
            ilha_df = df.groupby('Ilha_Balcao').size().reset_index(name='N')
            fig_ilha = px.bar(ilha_df, x='Ilha_Balcao', y='N', title='Clientes por Ilha', color='Ilha_Balcao', text='N')
            fig_ilha.update_traces(textposition='outside')
            st.plotly_chart(fig_ilha, use_container_width=True)
        with c2:
            st.plotly_chart(criar_grafico_risco_barras(df, 'Tipo_Cliente', 'Risco por Tipo de Cliente'), use_container_width=True)
            st.plotly_chart(px.pie(df, names='Tipo_Cliente', title='Distribuição por Tipo de Cliente', hole=0.3, color_discrete_map={'Residente': '#2ECC71', 'Emigrante': '#F39C12', 'Não Residente': '#E74C3C'}), use_container_width=True)
        
        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            df['Faixa_Etaria'] = pd.cut(df['Idade'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
            st.plotly_chart(criar_grafico_risco_barras(df, 'Faixa_Etaria', 'Risco por Faixa Etária', stacked=True), use_container_width=True)
        with c4:
            fig_score = px.histogram(df, x='Score_Conformidade', nbins=20, title='Distribuição do Score de Conformidade', color_discrete_sequence=['#3498DB'])
            fig_score.add_vline(x=df['Score_Conformidade'].mean(), line_dash="dash", annotation_text=f"Média: {df['Score_Conformidade'].mean():.1f}")
            st.plotly_chart(fig_score, use_container_width=True)
        
        st.divider()
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(criar_grafico_risco_barras(df, 'Ilha_Balcao', 'Risco por Ilha'), use_container_width=True)
        with c6: st.plotly_chart(criar_grafico_risco_barras(df, 'PEP_RCA', 'Risco por Status PEP/RCA'), use_container_width=True)
    
    # ========== ABA 2: MONITORIZAÇÃO KYC ==========
    with tab2:
        st.header("Monitorização KYC")
        f1, f2, f3 = st.columns(3)
        with f1: filtro_risco = st.multiselect("Nível de Risco", CONFIG['RISCO']['ordem'], default=CONFIG['RISCO']['ordem'])
        with f2: filtro_status = st.multiselect("Status", df['Status_Atualizacao'].unique(), default=['A vencer', 'Atrasado'])
        with f3: filtro_ilha = st.multiselect("Ilha", df['Ilha_Balcao'].unique(), default=df['Ilha_Balcao'].unique())
        
        df_filt = df[(df['Nivel_Risco'].isin(filtro_risco)) & (df['Status_Atualizacao'].isin(filtro_status)) & (df['Ilha_Balcao'].isin(filtro_ilha))].copy()
        df_filt['Dias_para_Vencer'] = (df_filt['Proxima_Data_Atualizacao'] - datetime(2024, 12, 1)).dt.days
        
        st.subheader("Clientes que Requerem Atenção")
        st.dataframe(df_filt[['ID_Cliente', 'Ilha_Balcao', 'Nivel_Risco', 'Tipo_Cliente', 'Status_Atualizacao', 'Proxima_Data_Atualizacao', 'Dias_para_Vencer', 'Score_Conformidade', 'PEP_RCA']].sort_values('Dias_para_Vencer'), use_container_width=True, hide_index=True)
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            df['Mes_Vencimento'] = df['Proxima_Data_Atualizacao'].dt.to_period('M').astype(str)
            timeline = df.groupby(['Mes_Vencimento', 'Ilha_Balcao']).size().reset_index(name='Count')
            st.plotly_chart(px.line(timeline, x='Mes_Vencimento', y='Count', color='Ilha_Balcao', title='Timeline de Vencimentos', markers=True), use_container_width=True)
        with c2:
            status_tipo = df.groupby(['Tipo_Cliente', 'Status_Atualizacao']).size().unstack(fill_value=0)
            fig = go.Figure([go.Bar(name=s, x=status_tipo.index, y=status_tipo[s], marker_color={'Em dia': '#2ECC71', 'A vencer': '#F39C12', 'Atrasado': '#E74C3C'}[s]) for s in ['Em dia', 'A vencer', 'Atrasado'] if s in status_tipo.columns])
            fig.update_layout(title='Status de Atualização por Tipo', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== ABA 3: GESTÃO DOCUMENTAL ==========
    with tab3:
        st.header("Gestão Documental")
        cols = st.columns(4)
        metricas_doc = [
            ("Total Documentos", f"{len(df):,}", None),
            ("% Válidos", f"{(df['Status_Documento'] == 'Válido').mean()*100:.1f}%", None),
            ("% Quase Expirando", f"{(df['Status_Documento'] == 'Quase Expirando').mean()*100:.1f}%", None),
            ("% Expirados", f"{(df['Status_Documento'] == 'Expirado').mean()*100:.1f}%", f"{(df['Status_Documento'] == 'Expirado').sum()} docs")
        ]
        for col, (t, v, d) in zip(cols, metricas_doc):
            with col: st.metric(t, v, delta=d)
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            doc_status = df.groupby(['Tipo_Documento', 'Status_Documento']).size().unstack(fill_value=0)
            fig = go.Figure([go.Bar(name=s, x=doc_status.index, y=doc_status[s], marker_color={'Válido': '#2ECC71', 'Quase Expirando': '#F39C12', 'Expirado': '#E74C3C'}[s]) for s in ['Válido', 'Quase Expirando', 'Expirado'] if s in doc_status.columns])
            fig.update_layout(title='Status por Tipo de Documento', barmode='stack', xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(df.groupby(['Tipo_Cliente', 'Tipo_Documento']).size().reset_index(name='Count'), x='Tipo_Cliente', y='Count', color='Tipo_Documento', title='Documentos por Tipo de Cliente', barmode='stack'), use_container_width=True)
        
        st.divider()
        st.subheader("Documentos Expirados")
        f1, f2 = st.columns(2)
        with f1: filtro_ilha_d = st.multiselect("Ilha", df['Ilha_Balcao'].unique(), default=df['Ilha_Balcao'].unique(), key='ilha_doc')
        with f2: filtro_tipo_d = st.multiselect("Tipo Cliente", df['Tipo_Cliente'].unique(), default=df['Tipo_Cliente'].unique(), key='tipo_doc')
        
        expirados = df[(df['Status_Documento'] == 'Expirado') & (df['Ilha_Balcao'].isin(filtro_ilha_d)) & (df['Tipo_Cliente'].isin(filtro_tipo_d))]
        st.dataframe(expirados[['ID_Cliente', 'Ilha_Balcao', 'Tipo_Cliente', 'Tipo_Documento', 'Data_Validade', 'Nivel_Risco', 'PEP_RCA']].sort_values('Data_Validade'), use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", expirados.to_csv(index=False).encode('utf-8'), f"expirados_{datetime.now():%Y%m%d}.csv", "text/csv")
    
    # ========== ABA 4: REDE BAYESIANA ==========
    with tab4:
        st.header("Rede Bayesiana")
        st.markdown("Modelagem probabilística com **Eliminação de Variáveis**")
        if modelo: st.plotly_chart(criar_grafo_rede(modelo), use_container_width=True)
        
        st.divider()
        st.subheader("Inferência Dinâmica")
        opcoes = ['Não informado', 'Não', 'Sim']
        opcoes_inc = ['Não informado', 'Nenhum', '1-2 incidentes', '3+ incidentes']
        opcoes_rec = ['Não informado', 'Baixo', 'Médio', 'Alto']
        opcoes_freq = ['Não informado', '5 anos', '2 anos', '1 ano']
        
        c1, c2, c3 = st.columns(3)
        with c1:
            ev_doc = st.selectbox("Documentação Expirada", opcoes)
            ev_atr = st.selectbox("Atraso na Atualização", opcoes)
            ev_nao_res = st.selectbox("Cliente Não Residente", opcoes)
        with c2:
            ev_pep = st.selectbox("PEP/RCA", opcoes)
            ev_inc = st.selectbox("Histórico de Incidentes", opcoes_inc)
            ev_rec = st.selectbox("Número de Reclamações", opcoes_rec)
        with c3:
            ev_score = st.selectbox("Score de Conformidade", opcoes_rec)
            ev_freq = st.selectbox("Frequência de Atualização", opcoes_freq)
        
        evidencias = {k: v for k, v in [
            ('Documentacao_Expirada', ev_doc), ('Atraso_Atualizacao', ev_atr),
            ('Cliente_Nao_Residente', ev_nao_res), ('PEP_RCA', ev_pep),
            ('Historico_Incidentes', ev_inc), ('Numero_Reclamacoes', ev_rec),
            ('Score_Conformidade', ev_score), ('Frequencia_Atualizacao', ev_freq)
        ] if v != 'Não informado'}
        
        if st.button("🔍 Calcular Probabilidades", type="primary") and evidencias and inferencia:
            prob = calcular_probabilidade_risco(inferencia, evidencias)
            if prob:
                cols = st.columns(3)
                for col, risco in zip(cols, CONFIG['RISCO']['ordem']):
                    with col:
                        st.metric(f"P(Risco = {risco})", f"{prob[risco]:.2%}", delta="⚠️ Atenção" if risco == 'Alto' and prob[risco] > 0.5 else None)
                fig = go.Figure(go.Bar(x=list(prob.keys()), y=list(prob.values()), marker_color=[CONFIG['RISCO']['cores'][r] for r in CONFIG['RISCO']['ordem']]))
                fig.update_layout(title='Distribuição de Probabilidade', yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== ABA 5: MATRIZ DE RISCO ==========
    with tab5:
        st.header("Matriz de Risco do Cliente - Modelo Paramétrico")
        modo = st.radio("Modo de avaliação:", ["Novo Cliente", "Cliente Existente"], horizontal=True)
        
        if 'matriz_inputs' not in st.session_state: 
            st.session_state.matriz_inputs = {}
        
        if modo == "Cliente Existente":
            col_busca, col_btn = st.columns([3, 1])
            with col_busca: 
                cliente_id = st.selectbox("Selecione o Cliente", df['ID_Cliente'].tolist(), key='sel_cliente')
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Carregar Dados", type="primary"):
                    cli = df[df['ID_Cliente'] == cliente_id].iloc[0]
                    st.session_state.matriz_inputs = {
                        'tipo': cli['Tipo_Cliente'], 
                        'rend': int(cli['Rendimento_Mensal']), 
                        'hist': cli['Historico_Incidentes'],
                        'doc': 'Sim' if cli['Status_Documento'] == 'Expirado' else 'Não', 
                        'freq': cli['Frequencia_Atualizacao'],
                        'tempo': int(cli['Tempo_Relacionamento']), 
                        'prod': int(cli['Numero_Produtos']),
                        'rec': 'Baixo (0)' if cli['Numero_Reclamacoes'] == 0 else 'Médio (1-2)' if cli['Numero_Reclamacoes'] <= 2 else 'Alto (3+)',
                        'pais': cli['Pais_Residencia'], 
                        'ativ': cli['Atividade_Economica'], 
                        'pep': cli['PEP_RCA'], 
                        'nac': cli['Multipla_Nacionalidade']
                    }
                    st.success(f"Cliente {cliente_id} carregado!")
        
        st.divider()
        st.subheader("Parâmetros de Avaliação")
        inp = st.session_state.matriz_inputs
        
        c1, c2, c3 = st.columns(3)
        with c1:
            tipo = st.selectbox("Tipo de Cliente *", ['Residente', 'Emigrante', 'Não Residente'], 
                               index=['Residente', 'Emigrante', 'Não Residente'].index(inp.get('tipo', 'Residente')))
            rend = st.number_input("Rendimento Mensal (CVE) *", 0, 1000000, inp.get('rend', 25000), 1000)
            st.caption(f"≈ €{rend/110:.0f} (taxa: 110 CVE/€)")
        with c2:
            doc = st.selectbox("Documento Expirado *", ['Não', 'Sim'], 
                              index=['Não', 'Sim'].index(inp.get('doc', 'Não')))
            freq = st.selectbox("Frequência Atualização *", ['5 anos', '2 anos', '1 ano'], 
                               index=['5 anos', '2 anos', '1 ano'].index(inp.get('freq', '5 anos')))
            tempo = st.number_input("Tempo Relacionamento (anos) *", 0, 50, inp.get('tempo', 5))
        with c3:
            rec = st.selectbox("Reclamações *", ['Baixo (0)', 'Médio (1-2)', 'Alto (3+)'], 
                              index=['Baixo (0)', 'Médio (1-2)', 'Alto (3+)'].index(inp.get('rec', 'Baixo (0)')))
            pais = st.text_input("País de Residência", inp.get('pais', 'Cabo Verde'))
            ativ = st.selectbox("Atividade Econômica", CONFIG['ATIVIDADES'], 
                               index=CONFIG['ATIVIDADES'].index(inp.get('ativ', 'Outra')) if inp.get('ativ') in CONFIG['ATIVIDADES'] else len(CONFIG['ATIVIDADES'])-1)
        
        c4, c5, c6 = st.columns(3)
        with c4:
            hist = st.selectbox("Histórico Incidentes *", ['Nenhum', '1-2 incidentes', '3+ incidentes'], 
                               index=['Nenhum', '1-2 incidentes', '3+ incidentes'].index(inp.get('hist', 'Nenhum')))
            prod = st.number_input("Número de Produtos *", 0, 20, inp.get('prod', 3))
        with c5: 
            pep = st.selectbox("PEP/RCA *", ['Não', 'Sim'], 
                              index=['Não', 'Sim'].index(inp.get('pep', 'Não')))
        with c6: 
            nac = st.text_input("Nacionalidade(s)", inp.get('nac', 'Cabo-Verdiana'))
        
        st.divider()
        
        if st.button("🧮 Calcular Score de Risco", type="primary"):
            score, classif, prob_alto, fatores = calcular_score_manual(tipo, rend, hist, doc, freq, tempo, prod, rec, pais, ativ, pep, nac)
            
            r1, r2, r3 = st.columns(3)
            
            # COLUNA 1: Gauge e Fatores
            with r1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=score, 
                    title={'text': "Score de Risco"}, 
                    gauge={
                        'axis': {'range': [0, 30]}, 
                        'bar': {'color': "darkblue"}, 
                        'steps': [
                            {'range': [0, 6], 'color': '#2ECC71'}, 
                            {'range': [6, 14], 'color': '#F39C12'}, 
                            {'range': [14, 30], 'color': '#E74C3C'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=350)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                if fatores:
                    st.markdown("**⚠️ Fatores de Risco:**")
                    for f in fatores[:5]:
                        st.write(f"• {f}")
                else:
                    st.success("✓ Nenhum fator crítico identificado")
            
            # COLUNA 2: Classificação e Recomendações
            with r2:
                # Exibe título com cor
                if classif == 'Baixo':
                    st.markdown("### 🟢 **RISCO BAIXO**")
                elif classif == 'Médio':
                    st.markdown("### 🟡 **RISCO MÉDIO**")
                else:
                    st.markdown("### 🔴 **RISCO ALTO**")
                
                # Recomendações
                st.markdown("**Recomendações:**")
                if classif == 'Baixo':
                    st.write("• Atualização KYC a cada 5 anos")
                    st.write("• Monitorização padrão")
                    st.write("• Documentação simplificada aceitável")
                    st.write("• Sem restrições adicionais")
                elif classif == 'Médio':
                    st.write("• Atualização KYC a cada 2 anos")
                    st.write("• Monitorização reforçada trimestral")
                    st.write("• Verificação periódica de documentação")
                    st.write("• Análise de movimentações atípicas")
                else:
                    st.write("• Atualização KYC anual obrigatória")
                    st.write("• Monitorização contínua mensal")
                    st.write("• Análise detalhada de transações")
                    st.write("• Reporte obrigatório ao compliance")
                    st.write("• Aprovação de gestor sénior")
                
                st.divider()
                st.write(f"**País:** {pais}")
                st.write(f"**Atividade:** {ativ}")
                st.write(f"**Nacionalidade:** {nac}")
            
            # COLUNA 3: Comparação com Bayesiano
            with r3:
                st.markdown("### Comparação com Modelo Bayesiano")
                
                # Prepara evidências
                evidencias = {
                    'Documentacao_Expirada': doc,
                    'Cliente_Nao_Residente': 'Sim' if tipo == 'Não Residente' else 'Não',
                    'PEP_RCA': pep,
                    'Historico_Incidentes': hist,
                    'Numero_Reclamacoes': 'Baixo' if rec == 'Baixo (0)' else 'Médio' if rec == 'Médio (1-2)' else 'Alto',
                    'Frequencia_Atualizacao': freq
                }
                
                if inferencia:
                    prob_bayes = calcular_probabilidade_risco(inferencia, evidencias)
                    if prob_bayes:
                        p_alto_bayes = prob_bayes.get('Alto', 0)
                        
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            st.metric("Bayesiano", f"{p_alto_bayes:.1%}")
                        with col_b2:
                            st.metric("Paramétrico", f"{prob_alto:.1%}")
                        
                        diff = abs(p_alto_bayes - prob_alto)
                        if diff < 0.15:
                            st.success(f"✓ Convergentes (Δ = {diff:.1%})")
                        else:
                            st.warning(f"⚠️ Divergência (Δ = {diff:.1%})")
                
                st.divider()
                
                # Comparação com a carteira
                pct_alto = (df['Nivel_Risco'] == 'Alto').mean()
                st.metric("% Risco Alto na Carteira", f"{pct_alto:.1%}")
                
                if prob_alto > pct_alto:
                    st.warning("📊 Cliente acima da média da carteira")
                else:
                    st.info("📊 Cliente na média ou abaixo")
                
                st.divider()
                st.markdown("**Distribuição da Carteira:**")
                
                # Exibe distribuição sem usar markdown com código
                for t, p in df['Tipo_Cliente'].value_counts(normalize=True).items():
                    if t == 'Não Residente':
                        st.write(f"🔴 {t}: {p:.1%}")
                    elif t == 'Emigrante':
                        st.write(f"🟡 {t}: {p:.1%}")
                    else:
                        st.write(f"🟢 {t}: {p:.1%}")

if __name__ == "__main__":
    main()