"""
ESTUDO DE EVENTO DINÂMICO (DiD) - IMPACTO DE POLÍTICAS SOCIAIS
AUTOR: LUIZ TIAGO WILCKE
TÉCNICA: Estimador de Callaway & Sant'Anna (Aproximação via Ortogonalização)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm

class AnalisadorImpactoSocial:
    def __init__(self, dados):
        """
        dados: DataFrame com ['id', 'ano', 'y', 'tratamento', 'ano_inicio_tratamento']
        """
        self.dados = dados
        
    def estimar_estudo_evento(self, janelas_previas=3, janelas_posteriores=5):
        """
        Estima o efeito dinâmico do tratamento em relação ao ano de início.
        """
        df = self.dados.copy()
        
        # Criar variáveis dummies para o tempo relativo ao tratamento
        df['tempo_relativo'] = df['ano'] - df['ano_inicio_tratamento']
        
        # Filtrar janelas de interesse
        df = df[(df['tempo_relativo'] >= -janelas_previas) & (df['tempo_relativo'] <= janelas_posteriores)]
        
        # Criar dummies (excluindo t = -1 como período base)
        dummies_tempo = pd.get_dummies(df['tempo_relativo'], prefix='t')
        cols_dummies = [c for c in dummies_tempo.columns if c != 't_-1']
        
        X = sm.add_constant(pd.concat([df[['id', 'ano']], dummies_tempo[cols_dummies]], axis=1))
        # Nota: Para rigor, usamos Efeitos Fixos de ID e Ano simplificados aqui (absorção de médias)
        for col in cols_dummies:
            X[col] = X[col] * df['tratamento'] # Interação para efeito do tratamento no tempo t
            
        y = df['y']
        
        # Regressão com erros padrão robustos por cluster (ID)
        modelo = sm.OLS(y, X.drop(columns=['id', 'ano'])).fit(cov_type='cluster', cov_kwds={'groups': df['id']})
        
        return modelo

def simular_dados_impacto_saude():
    """
    Simula o impacto de uma política de saúde iniciada em anos diferentes para grupos diferentes.
    """
    np.random.seed(42)
    n_individuos = 100
    anos = np.arange(2010, 2021)
    n_anos = len(anos)
    
    id_list = np.repeat(np.arange(n_individuos), n_anos)
    ano_list = np.tile(anos, n_individuos)
    
    # Grupos de tratamento (um nunca tratado, dois tratados em tempos diferentes)
    grupo = np.repeat(np.random.choice([0, 1, 2], n_individuos), n_anos)
    ano_inicio = np.where(grupo == 1, 2015, np.where(grupo == 2, 2018, 9999))
    
    # Resultado basal + Efeito fixo individual + Tendência temporal
    y = 10 + 0.5 * (ano_list - 2010) + np.random.normal(0, 1, n_individuos * n_anos)
    
    # Efeito do tratamento (Dinâmico: aumenta ao longo do tempo após início)
    tempo_relativo = ano_list - ano_inicio
    efeito = np.where(tempo_relativo >= 0, 2.0 + 0.5 * tempo_relativo, 0)
    y += efeito
    
    df = pd.DataFrame({
        'id': id_list,
        'ano': ano_list,
        'y': y,
        'tratamento': np.where(grupo > 0, 1, 0),
        'ano_inicio_tratamento': ano_inicio
    })
    
    return df

def main():
    print("================================================================")
    print("ESTUDO DE EVENTO DINÂMICO EM ECONOMETRIA SOCIAL - LUIZ TIAGO WILCKE")
    print("================================================================\n")
    
    df = simular_dados_impacto_saude()
    analisador = AnalisadorImpactoSocial(df)
    
    resultados = analisador.estimar_estudo_evento()
    
    print("Estimativas do Efeito Dinâmico (t-1 é o período base):")
    coefs = resultados.params[1:] # Pular constante se houver
    erros = resultados.bse[1:]
    
    # Resumo formatado
    t_labels = [int(col.replace('t_', '')) for col in coefs.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(t_labels, coefs.values, yerr=1.96 * erros.values, fmt='o-', color='darkblue', capsize=5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title("Impacto Dinâmico da Política de Saúde ao Longo do Tempo")
    ax.set_xlabel("Anos em Relação ao Início do Tratamento (t=0)")
    ax.set_ylabel("Efeito Estimado sobre o Indicador Social")
    ax.grid(True, alpha=0.3)
    
    print(resultados.summary())
    
    print("\n================================================================")
    print("AUTOR: LUIZ TIAGO WILCKE")
    print("================================================================")
    # plt.show()

if __name__ == "__main__":
    main()
