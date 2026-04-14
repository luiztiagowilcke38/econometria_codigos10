# INTERAÇÕES SOCIAIS E EFEITO PARES (PEER EFFECTS)
# AUTOR: LUIZ TIAGO WILCKE
# TÉCNICA: Identificação de Manski e Estimação GMM em Redes Sociais

library(Matrix)
library(matrixStats)

#' Estima o Modelo de Interacao Social Linear-em-Medias
#' Equacao: y = alpha + beta*W*y + gamma*X + delta*W*X + eps
#' Onde beta mede o efeito 'endogeno' (peer effect)
estimar_efeito_pares <- function(y, X, W) {
  n <- length(y)
  
  # Instrumentos de Kelejian e Robinson: W*X, W^2*X (para beta*W*y)
  # A endogeneidade vem de beta*W*y (y depende da media do grupo, que depende de y)
  WX <- W %*% X
  W2X <- W %*% WX
  
  # Matriz de Instrumentos
  H <- cbind(1, X, WX, W2X)
  
  # Primeiro Estagio: Projetar W*y nos instrumentos
  Wy <- W %*% y
  fit_1 <- lm(Wy ~ H - 1)
  Wy_hat <- fit_1$fitted.values
  
  # Segundo Estagio: Regressao Final
  # y = alpha + beta*Wy_hat + gamma*X + delta*WX + resid
  matriz_final <- cbind(1, Wy_hat, X, WX)
  colnames(matriz_final) <- c("Constante", "Efeito_Endogeno_Beta", "Efeito_Contextual_Gamma", "Efeito_Exogeno_WX")
  
  modelo_final <- lm(y ~ matriz_final - 1)
  
  return(summary(modelo_final))
}

# Simular rede de amizades e interacoes (Modelo de Peer Effects)
simular_interacoes_sociais <- function(n = 500, n_grupos = 50) {
  set.seed(123)
  
  # Criar matriz de adjacencia em blocos (grupos/salas de aula)
  n_per_grupo <- n / n_grupos
  W_list <- lapply(1:n_grupos, function(g) {
    m <- matrix(1/(n_per_grupo-1), n_per_grupo, n_per_grupo)
    diag(m) <- 0
    return(m)
  })
  W <- as.matrix(bdiag(W_list))
  
  # Covariaveis (ex: renda dos pais, esforco individual)
  X <- rnorm(n, mean = 10, sd = 2)
  
  # Parametros Reais
  alpha <- 5
  beta <- 0.6  # Efeito dos pares (social multiplier)
  gamma <- 0.8
  delta <- 0.2
  
  # y = (I - beta*W)^-1 * (alpha + gamma*X + delta*W*X + eps)
  identidade <- diag(n)
  A_inv <- solve(identidade - beta * W)
  
  eps <- rnorm(n, 0, 1)
  y <- A_inv %*% (alpha + gamma * X + delta * (W %*% X) + eps)
  
  return(list(y = as.vector(y), X = X, W = W))
}

# Codigo Principal
cat("================================================================\n")
cat("ECONOMETRIA DE INTERAÇÕES SOCIAIS - LUIZ TIAGO WILCKE\n")
cat("================================================================\n\n")

dados <- simular_interacoes_sociais()
resultado <- estimar_efeito_pares(dados$y, dados$X, dados$W)

print(resultado)

cat("\n================================================================\n")
cat("AUTOR: LUIZ TIAGO WILCKE - ANALISE DE IMPACTO SOCIAL\n")
cat("================================================================\n")
