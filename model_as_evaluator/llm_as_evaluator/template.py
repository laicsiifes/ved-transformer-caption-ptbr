# Evaluation prompt template based on G-Eval
EVALUATION_PROMPT_TEMPLATE = """
Você receberá uma descrição de imagem gerada por um modelo. Sua tarefa é avaliar a descrição gerada com base em uma métrica, comparando-a com descrições de referência consideras verdadeiras.
Certifique-se de ler e compreender estas instruções com atenção.
Mantenha este documento aberto durante a revisão e consulte-o sempre que necessário.

Critério de Avaliação:

{criteria}

Etapas de Avaliação:

{steps}

Exemplo:

Descrições de Referẽncia:

{reference_captions}

Descrição Gerada pelo Modelo:

{generated_caption}

Formulário de Avaliação (APENAS pontuações):

- {metric_name}
"""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevância (1-5) - seleção de conteúdo importante das descrições de referência. \
A descrição gerada deve incluir apenas informações importantes das descrições de referência. \
Os anotadores foram instruídos a penalizar descrições geradas que contivessem redundâncias e excesso de informações.
"""

RELEVANCY_SCORE_STEPS = """
1. Leia atentamente a descrição gerada e as descrições de referência.
2. Compare a descrição gerada com as descrições de referência e identifique os principais pontos abordados.
3. Avalie o quão bem a descrição gerada abrange os pontos principais das descrições de referência e quanta informação irrelevante ou redundante ela contém.
4. Atribua uma pontuação de relevância de 1 a 5.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coerência (1-5) - a qualidade dos conteúdos presentes na descrição gerada. \
Alinhamos esta dimensão com a questão de qualidade sobre estrutura e coerência, \
segundo a qual "a descrição gerada deve ser bem estruturada e bem organizada. \
A descrição gerada não deve ser apenas um amontoado de informações relacionadas, mas deve construir\
uma descrição gramaticalmente correta mantendo um conjunto coerente de informações relacionadas com\
as informações presentes nas descrições de referência".
"""

COHERENCE_SCORE_STEPS = """
1. Leia as descriçẽs de referência atentamente e identifique o tema principal e os pontos-chave.
2. Leia a descrição gerada e compare-a com as descrições de referência. Verifique se a descrição gerada abrange o tema principal e os pontos-chave das descrições de referência e se os apresenta em uma ordem clara e lógica.
3. Atribua uma pontuação de coerência em uma escala de 1 a 5, onde 1 é a nota mais baixa e 5 a mais alta, com base nos Critérios de Avaliação.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistência (1-5) - o alinhamento factual entre a descrição gerada e a as descrições de referência. \
Uma descrição gerada factualmente consistente contém apenas declarações que são implícitas nas descrições de referência. \
Os anotadores também foram solicitados a penalizar descrições geradas que continham fatos alucinatórios.
"""

CONSISTENCY_SCORE_STEPS = """
1. Leia as descrições de referência atentamente e identifique os principais fatos e detalhes que elas apresentam.
2. Leia a descrição gerada e compare-a com as descrições de referência. Verifique se a descrição gerada contém algum erro factual que não seja corroborado pelas descrições de referência.
3. Atribua uma pontuação de consistência com base nos Critérios de Avaliação.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the image description in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The image description has many errors that make it hard to understand or sound unnatural.
2: Fair. The image description has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The image description has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the image description and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""