"""
PROJETO: ECONOMETRIA AVANÇADA - CAUSAL MACHINE LEARNING E IV
AUTOR: LUIZ TIAGO WILCKE
DESCRIÇÃO: Implementação de Double Machine Learning (DML) com Cross-Fitting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class MotorDoubleMachineLearning:
    """
    Implementa o algoritmo de Double Machine Learning (Chernozhukov et al. 2018).
    Resolve o viés de seleção de variáveis em alta dimensão.
    """
    def __init__(self, y, D, X, n_folds=5):
        self.y = y
        self.D = D
        self.X = X
        self.n = len(y)
        self.n_folds = n_folds
        
    def estimar_efeito_causal(self):
        # 1. Preparar resíduos
        residuos_y = np.zeros(self.n)
        residuos_d = np.zeros(self.n)
        
        # 2. Cross-Fitting (K-Fold)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for treino_idx, teste_idx in kf.split(self.X):
            # Modelos nuisances (E[Y|X] e E[D|X])
            modelo_y = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
            modelo_d = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
            
            # Treinar no fold de treino
            modelo_y.fit(self.X[treino_idx], self.y[treino_idx])
            modelo_d.fit(self.X[treino_idx], self.D[treino_idx])
            
            # Calcular resíduos no fold de teste (ortogonalização)
            residuos_y[teste_idx] = self.y[teste_idx] - modelo_y.predict(self.X[teste_idx])
            residuos_d[teste_idx] = self.D[teste_idx] - modelo_d.predict(self.X[teste_idx])
            
        # 3. Estimação final (Regressão nos Resíduos com Erros Robustos)
        # Nota: Não incluímos constante pois os resíduos já têm média zero por construção.
        import statsmodels.api as sm
        modelo_final = sm.OLS(residuos_y, residuos_d).fit(cov_type='HC3')
        
        # Como não há constante, params[0] é o coeficiente do tratamento (theta)
        theta_causal = modelo_final.params[0]
        erro_padrao = modelo_final.bse[0]
        p_valor = modelo_final.pvalues[0]
        
        return theta_causal, erro_padrao, p_valor, residuos_y, residuos_d

def simular_dados_alta_dimensao(n=500, p=100, efeito_real=2.0):
    """
    Simula um dataset onde o tratamento D é endógeno (correlacionado com X).
    """
    np.random.seed(123)
    X = np.random.normal(0, 1, size=(n, p))
    # Apenas as primeiras 5 variáveis afetam o tratamento e o outcome
    coef_controle = np.exp(-np.arange(p)/10)
    D = X @ coef_controle + np.random.normal(0, 1, n)
    y = efeito_real * D + X @ (coef_controle * 2) + np.random.normal(0, 1, n)
    
    return y, D, X

def main():
    print("================================================================")
    print("ESTIMAÇÃO VIA DOUBLE MACHINE LEARNING (DML) - LUIZ TIAGO WILCKE")
    print("================================================================\n")
    
    # Simulação
    y, D, X = simular_dados_alta_dimensao()
    
    # Execução DML
    motor_dml = MotorDoubleMachineLearning(y, D, X)
    theta, erro_se, p_val, res_y, res_d = motor_dml.estimar_efeito_causal()
    
    print(f"Efeito Causal Real: 2.0000")
    print(f"Efeito Causal Estimado (DML): {theta:.4f}")
    print(f"Erro Padrão Robusto (HC3): {erro_se:.4f}")
    print(f"P-valor: {p_val:.4f}")
    
    # Comparação com OLS ingênuo (viesado)
    ols_ingenuo = LinearRegression().fit(np.column_stack([D, X]), y)
    print(f"Efeito OLS Ingênuo (incluindo controles): {ols_ingenuo.coef_[0]:.4f}")

    print("\nVisualizando a Ortogonalização dos Resíduos...")
    plt.figure(figsize=(10, 6))
    plt.scatter(res_d, res_y, alpha=0.5, color='darkred')
    plt.title("Regressão de Resíduos: Identificação do Efeito Causal")
    plt.xlabel("Resíduos do Tratamento (D - E[D|X])")
    plt.ylabel("Resíduos do Resultado (Y - E[Y|X])")
    plt.grid(True)
    # plt.show() # Omitido para automação, mas funcional

    print("\n================================================================")
    print("AUTOR: LUIZ TIAGO WILCKE")
    print("================================================================")

if __name__ == "__main__":
    main()
