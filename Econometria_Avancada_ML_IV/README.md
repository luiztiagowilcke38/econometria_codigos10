# Causal ML e Variáveis Instrumentais
**Autor:** Luiz Tiago Wilcke

O estado da arte em identificação causal robusta e teoria de momentos.

## Conteúdo
- **Double Machine Learning (DML):** Ortogonalização de Neyman e Cross-Fitting.
- **GMM Manual:** Estimador com matriz de pesos ótima e forma sanduíche.
- **Lasso/Ridge:** Seleção de controles em alta dimensionalidade.

## Aplicações Práticas em Econometria

### 1. Double Machine Learning (DML)
- **Avaliação de Políticas em Larga Escala:** Estimar o impacto de programas governamentais (ex: Bolsa Família ou subsídios industriais) onde existem milhares de controles potenciais (variáveis demográficas, geográficas, econômicas) e o risco de sobre-ajuste é alto.
- **Precificação e Elasticidade:** No setor privado (Industrial Organization), o DML permite estimar a elasticidade-preço da demanda de forma causal, limpando o viés de variáveis de confusão capturadas por modelos complexos.
- **Marketing Attribution:** Identificar o efeito incremental de campanhas de marketing em ambientes com múltiplos pontos de contato e alta dimensionalidade de dados.

### 2. GMM Manual e Variáveis Instrumentais
- **Estimativa de Modelos Estruturais:** Utilizado para estimar parâmetros de funções de produção ou equações de oferta/demanda onde o preço e a quantidade são determinados simultaneamente (equilíbrio).
- **Dados de Painel Dinâmico:** Aplicação do estimador de Arellano-Bond para modelos onde a variável dependente defasada é um regressor, comum em estudos de crescimento econômico e finanças corporativas.
- **Correção de Endogeneidade em Finanças:** Estimar a relação entre governança corporativa e valor de mercado, usando instrumentos para lidar com a causalidade reversa.

---
**Autor:** Luiz Tiago Wilcke
