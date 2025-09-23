# Evaluation prompt template based on CLAIR
EVALUATION_PROMPT_TEMPLATE="""
Você está tentando determinar se um conjunto de legendas candidato está descrevendo a mesma imagem que um conjunto de legendas de referência.
Conjunto de candidatos:
{candidate_statements}
Conjunto de referência:
{target_statements}
Em uma escala precisa de 0 a 100, qual a probabilidade de o conjunto de candidatos estar \
descrevendo a mesma imagem que o conjunto de referência? (Formato JSON, com uma chave "score",  \
um valor entre 0 e 100 e uma chave "reason" com um valor de string.)
"""