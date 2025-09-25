# Evaluation prompt template based on FLEUR
EVALUATION_PROMPT_TEMPLATE="""
Sua tarefa é avaliar e classificar a descrição em uma escala de 0 a 100 com base \
nos Critérios de Avaliação fornecidos. 

**Critérios de Avaliação:**

0: A descrição não descreve a imagem.
100: A descrição descreve a imagem com precisão e clareza.

**Descrição:** {candidate_statements}

Imprimir **SOMENTE** formato JSON, com uma chave "score",  \
um valor de 0 a 100 e uma chave "reason" com um valor string descrevendo o seu raciocínio para \
o valor de score que você atribuiu.
"""