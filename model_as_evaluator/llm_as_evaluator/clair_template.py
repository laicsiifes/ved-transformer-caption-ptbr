# Evaluation prompt template based on CLAIR
EVALUATION_PROMPT_TEMPLATE="""
Você está tentando determinar se uma descrição candidato está descrevendo a mesma \
imagem que um conjunto de descrições de referência.

**Descrição candidata:**
{candidate_statements}

**Descrições de referência**:
{target_statements}

Em uma escala precisa de 0 a 100, qual a probabilidade da descrição candidata estar \
descrevendo a mesma imagem que o conjunto de referência? (Imprimir **SOMENTE** formato JSON, \
com uma chave "score", um valor entre 0 e 100 e uma chave "reason" com um valor string \
descrevendo o seu raciocínio para o valor de score que você atribuiu.)
"""