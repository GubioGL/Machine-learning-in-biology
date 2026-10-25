# Relatório de pesquisa: ML em biologia, diagnóstico dos dados e plano de análise

Documento de trabalho (outubro de 2026). Reúne a revisão da literatura de alto impacto, o que foi
encontrado na pasta do Drive *ML for biology*, o diagnóstico dos dados, a lista de algoritmos e as
mudanças feitas em relação ao primeiro rascunho (*Assinaturas perdidas.docx*). O artigo está em
`paper/main.tex`; a explicação didática de cada algoritmo está em `paper/guia_ponte_ml_biologia.tex`.

## Resumo

- O projeto é publicável, mas a mensagem central mudou de "o Random Forest provou a homogeneização
  do Pantanal" para "classificadores validados medem o quanto a assembleia de uma área úmida
  antropizada deixou de ser distinguível de uma savana seca".
- O teste principal agora é direto: comparar a separabilidade (AUC) de Pantanal × Cerrado com a de
  Pantanal × várzea, com bootstrap pareado, sempre contra um modelo nulo e testes de permutação.
- Na execução de outubro/2026: Pantanal × várzea AUC ≈ 1,00; Pantanal × Cerrado AUC ≈ 0,56 (IC
  inclui 0,5). O resultado se mantém em todos os cenários de robustez.
- Três limitações de desenho precisam aparecer no artigo: uma área por paisagem; Pantanal e Cerrado
  ~224 km um do outro e na mesma bacia (várzea a ~590 km, bacia do Paraná); campanhas diferentes
  (Cerrado set/2025, Pantanal quase todo nov/2025, várzea 2024).

## 1. Panorama da literatura de alto impacto

**Marcos do ML em biologia**

- *Estrutura de proteínas.* O AlphaFold (Jumper et al., Nature 596:583–589, 2021) atingiu no CASP14
  precisão comparável à de estruturas experimentais na maioria dos casos; o Nobel de Química de 2024
  foi dividido entre David Baker (desenho computacional de proteínas) e Demis Hassabis e John Jumper
  (predição de estrutura). <https://doi.org/10.1038/s41586-021-03819-2>
- *Modelos de fundação genômicos.* Evo 2 (Brixi et al., Nature, 2026) modela e projeta sequências
  de DNA de todos os domínios da vida. <https://doi.org/10.1038/s41586-026-10176-5>
- *Síntese.* Wang et al. (Nature 620:47–60, 2023) organizam o papel da IA na descoberta científica.
  <https://doi.org/10.1038/s41586-023-06221-2>

**Ecologia, entomologia e biodiversidade (mais próximos do nosso tema)**

- Norouzzadeh et al. (PNAS 115:E5716, 2018): redes convolucionais identificam animais em milhões de
  imagens de armadilhas fotográficas com acurácia próxima à de voluntários.
  <https://doi.org/10.1073/pnas.1719367115>
- Høye et al. (PNAS 118:e2002545117, 2021): visão computacional para monitoramento contínuo de insetos.
  <https://doi.org/10.1073/pnas.2002545117>
- Tuia et al. (Nature Communications 13:792, 2022): como aproximar ecólogos e cientistas de dados;
  vieses, generalização para novas regiões e interpretabilidade.
  <https://doi.org/10.1038/s41467-022-27980-y>
- Pichler & Hartig (Methods Ecol. Evol. 14:994–1016, 2023): em dados ecológicos tabulares os
  métodos de árvores (florestas, boosting) seguem dominantes; deep learning compensa com imagens,
  sons e dados massivos. Apoia a escolha de métodos de árvore para a nossa tabela.
  <https://doi.org/10.1111/2041-210X.14061>
- Homogeneização biótica em alto impacto: Aulus-Giacosa, Ollier & Bertelsmeier (Nature Communications
  15:2266, 2024) mostram que formigas não nativas tornam assembleias mais parecidas no mundo todo.
  <https://doi.org/10.1038/s41467-024-46359-9>
- Declínio de insetos: Wagner et al. (PNAS 118:e2023989118, 2021). Os números de van Klink et al.
  (Science 368:417–420, 2020) estão sob *Editorial Expression of Concern* da Science (jan/2026);
  se forem citados, citar com essa ressalva.

**Rigor metodológico (o que os revisores vão cobrar)**

- REFORMS (Kapoor et al., Science Advances 10:eadk3452, 2024): checklist de consenso para ciência
  baseada em ML. <https://doi.org/10.1126/sciadv.adk3452>
- DOME (Walsh et al., Nature Methods 18:1122–1127, 2021): Dados, Otimização, Modelo, Avaliação;
  recomenda uma tabela-resumo de ML (está no Apêndice do artigo).
  <https://doi.org/10.1038/s41592-021-01205-4>
- Ploton et al. (Nature Communications 11:4540, 2020): validação não espacial superestimou muito o
  poder preditivo de mapas de biomassa; com validação espacial ele praticamente sumiu.
  <https://doi.org/10.1038/s41467-020-18321-y>
- Roberts et al. (Ecography 40:913–929, 2017): validação cruzada em blocos para dados com estrutura
  espacial, temporal ou hierárquica. <https://doi.org/10.1111/ecog.02881>
- Strobl et al. (BMC Bioinformatics 8:25, 2007) e Altmann et al. (Bioinformatics 26:1340–1347, 2010):
  vieses da importância do RF e correção por permutação.
- Lundberg et al. (Nature Machine Intelligence 2:56–67, 2020): TreeSHAP.
- Valavi et al. (Ecography 44:1731–1742, 2021): RF padrão vai mal com classes desbalanceadas;
  down-sampling por árvore melhora. <https://doi.org/10.1111/ecog.05615>

## 2. O que há na pasta do Drive

- O link enviado (`1MmwpKCFYL-l515yl3QD990sFCfMeo4FN`) não abriu para a conta conectada; foi usada a
  pasta *ML for biology* da conta gubiofisika@gmail.com, que tem o conteúdo descrito.
- `dados_.csv.csv`: 653 indivíduos, 23 espécies, três áreas, três métodos (armadilha suspensa,
  Malaise, puçá). Ver `data/README.md`.
- `Assinaturas perdidas.docx`: rascunho com RF (ranger, 1000 árvores, mtry 4), OOB 73,9%, Kappa
  0,59, matriz de confusão, MDA + Altmann, PDP e Tabela S1.
- `MARKDOWN/`: 10 artigos convertidos (Leite et al. 2026; Breiman 2001; Cutler 2007; Valavi 2021;
  Ogan 2022; Fisogni 2025; Jacques 2023; De Castro 2021; Francisco 2023; um sem identificação).
- Manuscrito irmão *Not all paper dissolves in water* (Leite et al., em preparação): PERMANOVA de
  Jaccard F = 11,77, R² = 0,354; Pantanal × Cerrado não difere (p ajustado ≈ 0,087).
  **Nossa implementação em Python reproduz F = 11,77 e R² = 0,354** (p Pan × Cer = 0,085).

## 3. Diagnóstico dos dados (o que muda a interpretação)

1. **n pequeno**: 46 pontos, 17 espécies; intervalos largos.
2. **Desbalanceamento**: Pantanal 11 pontos, contra 18 e 17. O RF sem correção empurra o Pantanal
   para as classes maiores, o que imita a homogeneização. Com o RF balanceado a confusão
   Pantanal→Cerrado cai de ~65% para ~56% e passa a ser mútua (Cerrado→Pantanal sobe de ~13% para
   ~36%). A sobreposição é sinal; a assimetria ("o Cerrado absorveu o Pantanal") do rascunho era
   artefato do tamanho das classes.
3. **A várzea é quase disjunta**: nas armadilhas suspensas ela compartilha 1 espécie com o Pantanal
   e 2 com o Cerrado (Jaccard médio 1,00 e 0,99). A identidade da várzea é muito forte, quase trivial.
4. **A convergência Pantanal–Cerrado depende de poucas espécies**: *Agelaia bequaerti* está em 91%
   dos pontos do Pantanal e 82% do Cerrado; é a única espécie em 4 dos 11 pontos do Pantanal e 9 dos
   17 do Cerrado. Só *Polybia ruficeps* (4 pontos) é exclusiva do Pantanal.
5. **Uma área por paisagem** (pseudo-replicação): o modelo distingue áreas, não tipos de paisagem.
6. **Geografia**: Pantanal–Cerrado 224 km, mesma bacia (Paraguai); várzea 592 km do Pantanal, bacia
   do Paraná. Decaimento da similaridade com a distância é explicação alternativa.
7. **Campanhas**: Cerrado set/2025; Pantanal mai/2024 (2 pontos) e nov/2025 (9); várzea jan, mar e
   jun/2024. Dentro da várzea a variação entre campanhas é pequena (R² ≈ 0,10, n.s.), mas o efeito de
   campanha não pode ser testado para o par Pantanal–Cerrado.
8. **Pontos sem captura** não aparecem no CSV — confirmar com o grupo.
9. **Reconciliação de *A. bequaerti***: o IndVal (manuscrito irmão) a põe no Pantanal; o PDP do RF
   não ponderado (rascunho) a põe no Cerrado; no RF balanceado ela sobe as duas. Conclusão: ela marca
   o conjunto Pantanal+Cerrado, e a "afinidade" entre os dois depende do peso das classes.
10. **Referências do rascunho**: algumas tinham dados errados (ex.: Ogan et al. 2022 é pp. 1795–1809).
    O `paper/references.bib` foi conferido.

## 4. Lista de algoritmos

| # | Algoritmo | Pergunta biológica | Papel | Script |
|---|---|---|---|---|
| 0 | Classificador nulo (classe majoritária) | quanto se acerta sem biologia? | referência | `run/02` |
| 1 | Jaccard + PCoA + PERMANOVA + PERMDISP | as composições diferem? | descritivo | `run/01` |
| 2 | Regressão logística multinomial L2 | um modelo linear basta? pesos por espécie | baseline interpretável | `run/02` |
| 3 | k-NN com Jaccard | o que a semelhança de composição implica? | ponte ML–ecologia | `run/02` |
| 4 | Random Forest não ponderado | replicação do rascunho | baseline | `run/02` |
| 5 | Random Forest ponderado por classe | classificação com interações, sem viés de classe | principal | `run/02` |
| 6 | Balanced Random Forest | idem, down-sampling por árvore | **principal (pré-especificado)** | `run/02`–`05` |
| 7 | Gradient boosting raso | o resultado depende do algoritmo? | robustez | `run/02` |
| 8 | Classificadores por par + AUC + bootstrap pareado | **H2: AUC Pan×Cer < AUC Pan×Fpl?** | **teste central** | `run/03` |
| 9 | Probabilidades fora da amostra (simplex) | para onde o Pantanal "puxa"? | descritivo | `run/03` |
| 10 | Importância por permutação fora da amostra + Altmann | quais espécies sustentam a separação? | explicar | `run/04` |
| 11 | Dependência parcial e SHAP | direção do efeito de cada espécie | explicar | `run/04` |
| 12 | IndVal.g | visão clássica de indicadoras | ponte | `run/04` |
| 13 | CV com blocos espaciais, n igual, sem singletons, sem *A. bequaerti*, hiperparâmetros, campanhas | a conclusão é robusta? | validar | `run/05` |

Possíveis em versões futuras (detalhes no guia-ponte): conditional inference forests, `gllvm`,
`mvabund::manyglm`, HMSC, GDM, modelos de ocupação, rarefação/iNEXT. Deep learning não é
recomendado com n = 46.

**Validação comum a todos os modelos**: CV estratificada 5-fold × 50 repetições com as mesmas
partições para todos (comparação pareada); métricas nas predições fora da amostra de cada
repetição; hiperparâmetros fixados a priori (sem tuning, dispensa CV aninhada); testes de
permutação dos rótulos; relato segundo REFORMS/DOME.

## 5. Mudanças em relação ao rascunho

| Rascunho | Versão atual |
|---|---|
| RF em R, um modelo, OOB | 7 modelos em Python, CV repetida, nulo e permutação |
| H2 lida da matriz de confusão | H2 testada por AUC por pares com bootstrap pareado |
| desbalanceamento ignorado | RF ponderado e Balanced RF; efeito do balanceamento reportado |
| MDA no ajuste completo | importância fora da amostra + teste de Altmann |
| "comprova", "corrobora integralmente" | "compatível com"; limitações de desenho explícitas |
| números digitados à mão; p-valores sumidos | números gerados pelos scripts (`\res{}`) |
| sem robustez | 9 cenários + checagens de campanha |
| `Leite et al. [ano]` | `leite_companion` no `.bib` (substituir pela referência publicada) |

## 6. Próximos passos para os autores

1. Rodar `python run/run_all.py` e conferir os números com `data/results/`.
2. Biólogos: preencher os `\TODO` de descrição das áreas, antropização da Baía Negra, protocolo de
   campo e o parágrafo de mecanismos da Discussão.
3. Decidir a revista e trocar o `\documentclass`.
4. Depositar os dados (Zenodo) e só então tirar `data/raw/` do `.gitignore`.
5. Declarar o uso de ferramentas de IA (o rascunho citava o Gemini; incluir as demais usadas).
