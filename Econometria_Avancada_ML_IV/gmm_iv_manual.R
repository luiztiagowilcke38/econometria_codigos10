#' ---
#' Título: Estimação Manual via Método dos Momentos Generalizado (GMM)
#' Autor: Luiz Tiago Wilcke
#' Descrição: Implementação manual de GMM em dois passos para lidar
#'            com endogeneidade e heterocedasticidade.
#' ---

# Carregamento de dependências
library(matrixStats)

# ==============================================================================
# MOTOR GMM: ESTIMAÇÃO EM DOIS PASSOS
# ==============================================================================

estimar_gmm_manual <- function(y, X, Z) {
  cat("\nIniciando Estimação GMM em Dois Passos...\n")
  
  # 1. Primeiro Passo: Matriz de Pesos Identidade (Equivalente a IV simples)
  W1 <- diag(ncol(Z))
  
  # Função para o estimador GMM dado W
  calc_beta_gmm <- function(W) {
    # Beta = (X'Z * W * Z'X)^-1 * (X'Z * W * Z'y)
    termo_xz <- t(X) %*% Z
    inv_final <- solve(termo_xz %*% W %*% t(termo_xz))
    beta <- inv_final %*% (termo_xz %*% W %*% t(Z) %*% y)
    return(beta)
  }
  
  beta_passo1 <- calc_beta_gmm(W1)
  cat("Beta (Passo 1 - Pesos Identidade):", as.vector(beta_passo1), "\n")
  
  # 2. Resíduos do Primeiro Passo para construir a Matriz de Pesos Ótima
  residuos1 <- y - X %*% beta_passo1
  
  # Construção da matriz S (S = 1/n * sum(ui^2 * zi * zi')) para robustez
  n <- nrow(X)
  S <- matrix(0, ncol(Z), ncol(Z))
  for(i in 1:n) {
    S <- S + (residuos1[i]^2 * (Z[i, ] %*% t(Z[i, ])))
  }
  S <- S / n
  
  # Matriz de Pesos Ótima W2 = S^-1
  W2 <- solve(S)
  
  # 3. Segundo Passo: Estimação com Pesos Ótimos
  beta_gmm_final <- calc_beta_gmm(W2)
  
  # 4. Cálculo da Variância-Covariância Robusta (Forma Sanduíche)
  # Var = A^-1 * (X'Z * W * S * W * Z'X) * A^-1
  # Onde A = (X'Z * W * Z'X)
  termo_xz <- t(X) %*% Z
  A <- termo_xz %*% W2 %*% t(termo_xz)
  A_inv <- solve(A)
  
  # Miolo do sanduíche (B)
  B <- termo_xz %*% W2 %*% S %*% W2 %*% t(termo_xz)
  
  v_cov_gmm <- A_inv %*% B %*% A_inv
  erros_padrao <- sqrt(diag(v_cov_gmm))
  
  return(list(coeficientes = beta_gmm_final, erros = erros_padrao, matriz_pesos = W2))
}

# ==============================================================================
# SIMULAÇÃO E TESTE
# ==============================================================================

set.seed(42)
n_obs <- 1000

# Variável de Controle (X_exog), Instrumento (Z_inst) e Erro
z_inst <- rnorm(n_obs)
erro_u <- rnorm(n_obs)
x_exog <- rnorm(n_obs)

# Endogeneidade: X_endo depende do instrumento e do erro
x_endo <- 0.5 * z_inst + 0.3 * erro_u + rnorm(n_obs, sd=0.5)

# Equação de Interesse: Y = 2*X_endo + 1*X_exog + erro_u
y <- 2 * x_endo + 1 * x_exog + erro_u

# Preparar Matrizes (X inclui constante)
X_matriz <- cbind(1, x_endo, x_exog)
Z_matriz <- cbind(1, z_inst, x_exog)

# Executar GMM
resultado_final <- estimar_gmm_manual(y, X_matriz, Z_matriz)

cat("\n================================================================\n")
cat("RESULTADOS FINAIS - GMM SEGUNDO PASSO\n")
cat("Coeficientes:", as.vector(resultado_final$coeficientes), "\n")
cat("Erros Padrão:", as.vector(resultado_final$erros), "\n")
cat("================================================================\n")
cat("AUTOR: LUIZ TIAGO WILCKE\n")
cat("================================================================\n")
