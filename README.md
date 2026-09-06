# classificador-imagens-biomedicas

Classificação de pneumotórax em radiografias de tórax **sem treinar modelo
nenhum**: cada imagem DICOM vira um histograma de intensidade e é classificada
pela semelhança com as outras, comparando as quatro métricas de histograma do
OpenCV numa avaliação *leave-one-out* sobre 30 exames.

> **In English, briefly.** A pneumothorax classifier for chest radiographs that
> deliberately learns nothing. Each DICOM image is reduced to a normalized
> intensity histogram, and an unseen image is labelled by comparing its histogram
> against the other 29 — under all four OpenCV histogram metrics (correlation,
> chi-square, intersection, Bhattacharyya), each with its own image size, bin
> count, contrast pre-processing and decision rule. With 30 images (15 per class)
> there is no honest train/validation/test split, so evaluation is leave-one-out
> over the whole set. The point of the project is the comparison itself: how far
> a global, one-dimensional descriptor can go on a problem whose signal is local.
> Best recorded run: intersection, 0.73 sensitivity / 0.80 specificity — read the
> Limitations section before quoting that number.

---

## O problema

O **pneumotórax** é ar acumulado entre o pulmão e a parede torácica. Na
radiografia ele aparece como uma região sem trama vascular, quase sempre num
canto do campo pulmonar, delimitada por uma linha fina — a pleura visceral
deslocada. É um achado **local e de baixo contraste**.

O classificador deste repositório, de propósito, só enxerga estatística
**global**: a distribuição de tons de cinza da imagem inteira, sem nenhuma noção
de onde cada tom está. A pergunta que o projeto responde é justamente essa —
*quanto* de um achado local sobrevive a um descritor que joga fora toda a
informação espacial. A resposta, medida, está em [Resultados](#resultados-registrados):
melhor que sorteio, longe de utilizável.

---

## Os dados

30 imagens **DICOM**, 15 por classe, versionadas no próprio repositório em
`train/Pneumothorax/` e `train/No Pneumothorax/`. O que os cabeçalhos DICOM dizem
sobre elas:

| Característica | Valor |
|---|---|
| Modalidade | `CR` — radiografia computadorizada, região `CHEST` |
| Resolução | 1024 × 1024, 8 bits, `MONOCHROME2` |
| Compressão | JPEG baseline (`1.2.840.10008.1.2.4.50`) |
| Incidência | 10 PA + 5 AP em **cada** classe |
| Pacientes | 30 identificadores distintos — nenhum paciente aparece duas vezes |
| Anonimização | nome e ID substituídos por UUID, `StudyDate` zerada em `19010101`, sem instituição nem médico responsável. **Idade e sexo permanecem** |

A composição não é neutra e vale registrar antes de olhar qualquer métrica: a
idade média é **42,7 anos na classe positiva** contra **51,9 na negativa**, e o
sexo está distribuído 9F/6M contra 7F/8M. Com 15 imagens por classe, qualquer
diferença sistemática de porte, penetração do raio ou incidência entre os dois
grupos vira sinal para um classificador que só lê histograma — sem que ele tenha
visto pneumotórax nenhum. A paridade de incidência (10 PA / 5 AP dos dois lados)
foi o único desses eixos que ficou controlado.

**Origem do conjunto: o repositório não a declara.** Os arquivos chegaram já
anonimizados por DCMTK e renomeados em sequência; os cabeçalhos não apontam para
uma fonte nomeada. Ver [Sobre as imagens](#sobre-as-imagens-privacidade-e-licença).

A lista de arquivos de cada classe fica em `Pneumothorax_files.txt` e
`No_Pneumothorax_files.txt`, um caminho **relativo** por linha. É esse par de
arquivos, e não a varredura do diretório, que define o conjunto — trocar o
experimento é editar duas listas de texto, não mover imagem de pasta.

---

## Como funciona

```
   000001.dcm  ──► pydicom (pixel_array, JPEG → Pillow)
                        │
                        ▼
              normaliza min-max → uint8 0..255
                        │
                        ▼
              redimensiona (INTER_AREA) para 96², 128² ou 64²
                        │
                        ▼
              pré-processa: CLAHE (clip 2.0, tiles 8×8) ou nada
                        │
                        ▼
              histograma de N bins, normalizado em L1  ← a "assinatura" da imagem
                        │
                        ▼
   cv2.compareHist contra as outras 29 assinaturas
                        │
                        ▼
              regra de decisão → Pneumothorax / No Pneumothorax
```

Não há treino, não há pesos, não há estado salvo. `python main.py` reprocessa as
30 imagens do zero a cada execução — quatro vezes, uma por métrica, porque cada
métrica tem sua própria configuração de pré-processamento.

**As três regras de decisão** implementadas (`nearest`, `class_mean`,
`class_top3`) são o que transforma uma lista de distâncias em um rótulo:

| Regra | Decide por |
|---|---|
| `nearest` | o rótulo da **única** imagem mais parecida — vizinho mais próximo, k=1 |
| `class_mean` | a **média** da semelhança contra as 15 (ou 14) de cada classe |
| `class_top3` | a média das **3 melhores** de cada classe — meio-termo entre as duas |

A direção da comparação não é a mesma nas quatro métricas: correlação e
interseção querem valor **alto**, qui-quadrado e Bhattacharyya querem valor
**baixo**. Isso está no código como a flag `higher_is_better` de cada
`HistMethod`, e é o detalhe que faz `argmax`/`argmin` e os desempates de
`class_mean` continuarem corretos quando se troca de métrica — errar esse sinal
produz um classificador que roda, não quebra e acerta menos que sorteio.

---

## A decisão difícil: avaliar 30 imagens sem se enganar

Com 30 exemplos, separar um conjunto de teste é o caminho mais rápido para um
número sem significado: 20% viram 6 imagens, e uma única imagem trocada move a
acurácia em 17 pontos. A escolha aqui foi **leave-one-out** — cada imagem é
testada uma vez contra as outras 29, e a matriz de confusão soma as 30 rodadas.
Custa 30 avaliações completas em vez de uma, o que é irrelevante nessa escala, e
usa cada imagem como teste sem nunca deixá-la entre os candidatos da própria
predição (`samples[:idx] + samples[idx+1:]`).

Como cada uma das 30 imagens é de um paciente diferente, não existe o vazamento
clássico desse desenho — o mesmo paciente aparecendo dos dois lados da divisão.

O que o leave-one-out **não** resolve, e o repositório não resolve: a
configuração por métrica (tamanho, bins, pré-processamento, regra) foi escolhida
olhando para essas mesmas 30 imagens. Não há conjunto separado onde essa escolha
possa ser cobrada. Ver [Limitações](#limitações-conhecidas).

---

## Configuração por métrica

Cada métrica recebe seu próprio pré-processamento — a escolha que está gravada em
`METHOD_CONFIGS`, no `main.py`:

| Métrica | Imagem | Bins | Pré-processo | Regra |
|---|---|---|---|---|
| `CV_COMP_CORREL` | 96 × 96 | 32 | CLAHE | `class_mean` |
| `CV_COMP_CHISQR` | 128 × 128 | 64 | CLAHE | `class_mean` |
| `CV_COMP_INTERSECT` | 64 × 64 | 256 | CLAHE | `nearest` |
| `CV_COMP_BHATTACHARYYA` | 64 × 64 | 32 | nenhum | `class_mean` |

Duas coisas que a tabela mostra e o código confirma:

- **Bins e resolução andam em sentidos opostos.** A interseção usa a menor imagem
  (64²) com o histograma mais fino (256 bins, um por nível de cinza); o
  qui-quadrado usa a maior imagem (128²) com histograma grosso. Como o histograma
  é normalizado em L1, o tamanho da imagem não muda a escala do vetor — ele muda
  o quanto o `INTER_AREA` borrou a textura antes de contar.
- **O CLAHE é a equalização de contraste que sobrou.** O código implementa três
  modos (`raw`, `equalize`, `clahe`), mas a equalização global de histograma não
  é usada por nenhuma configuração — o que faz sentido no limite: equalizar o
  histograma global e depois comparar histogramas globais aproxima todas as
  imagens de uma mesma distribuição, apagando exatamente o que ia ser comparado.
  O CLAHE, sendo local e com clip, não tem esse efeito. O modo continua no código
  como opção; `class_top3` também está implementado e não é usado por nenhuma
  configuração.

---

## Resultados registrados

Os números abaixo são **os que estão versionados** em `metricas.txt`, saída de
uma execução do `main.py`. Classe positiva: `Pneumothorax`. Não foram
reexecutados para escrever este README.

| Métrica | TP | TN | FP | FN | Sensibilidade | Especificidade |
|---|---:|---:|---:|---:|---:|---:|
| `CV_COMP_CORREL` | 10 | 10 | 5 | 5 | 0,6667 | 0,6667 |
| `CV_COMP_CHISQR` | 7 | 13 | 2 | 8 | 0,4667 | 0,8667 |
| `CV_COMP_INTERSECT` | **11** | 12 | 3 | 4 | **0,7333** | 0,8000 |
| `CV_COMP_BHATTACHARYYA` | 8 | 13 | 2 | 7 | 0,5333 | 0,8667 |

Somando os acertos (o conjunto é balanceado, 15 por classe, então `(TP+TN)/30` é
lido direto): interseção **23/30**, Bhattacharyya **21/30**, correlação e
qui-quadrado **20/30**. O sorteio justo daria 15/30.

Lendo com honestidade:

- **A interseção é a única que não troca sensibilidade por especificidade.** As
  outras três acertam melhor o negativo do que o positivo — o qui-quadrado chega
  a 0,4667 de sensibilidade, ou seja, **erra mais da metade dos pneumotórax** e
  ainda assim exibe 20/30 de acerto total. Num achado clínico, esse é o erro que
  custa caro, e é o que a acurácia sozinha esconde.
- **O melhor resultado ainda deixa 4 dos 15 pneumotórax passar.** Não é um número
  de triagem; é a medida de até onde um descritor global chega num achado local.
- Sete das trinta decisões separam a interseção do qui-quadrado. Com n = 30, essa
  diferença é da ordem do ruído amostral — a tabela ordena as métricas, mas não
  prova que a ordem se manteria em outro conjunto de 30.

---

## Como rodar

Requer Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

A saída é o bloco por métrica — matriz de confusão, sensibilidade,
especificidade e a configuração usada — no mesmo formato de `metricas.txt`, o
que permite comparar direto com o resultado versionado.

Não há argumento de linha de comando: o `main.py` resolve tudo a partir do
diretório do próprio arquivo, e as imagens já estão no repositório. Alterar o
experimento é editar `METHOD_CONFIGS` ou as duas listas `.txt`.

**Sobre o `pillow` no `requirements.txt`:** ele não é importado por nenhuma linha
do código. Está lá porque as imagens são DICOM com pixel data comprimido em JPEG,
e o `pydicom` delega essa descompressão a um handler externo — sem Pillow (ou
GDCM), `ds.pixel_array` falha. Remover a dependência "não usada" quebra a
leitura das imagens.

---

## Limitações conhecidas

- **Não há conjunto de teste independente.** A configuração de cada métrica foi
  escolhida contra as mesmas 30 imagens sobre as quais o leave-one-out é
  calculado. As métricas da tabela são, portanto, otimistas por construção: elas
  descrevem o desempenho *neste* conjunto, não uma estimativa de generalização.
- **n = 30.** Um acerto a mais ou a menos move a sensibilidade em 6,7 pontos.
  Nenhuma comparação entre as quatro métricas aqui tem significância estatística,
  e o README não afirma que tenha.
- **Descritor cego a posição.** O histograma descarta *onde* está cada tom. Dois
  exames com a mesma distribuição de cinzas e anatomias completamente diferentes
  são idênticos para este classificador. É uma limitação escolhida, não um
  descuido — mas é o teto do método.
- **Confundidores não controlados.** Idade média e proporção de sexo diferem
  entre as classes (ver [Os dados](#os-dados)). Parte do acerto medido pode vir
  daí, e o desenho atual não separa uma coisa da outra.
- **Uma execução, não uma distribuição.** `metricas.txt` é o resultado de uma
  rodada. O pipeline é determinístico — não há sorteio nem inicialização
  aleatória —, então repetir a execução reproduz os mesmos números; o que não
  existe é variação sobre *conjuntos* diferentes, que é o que daria barra de erro.
- **Sem testes automatizados** e sem validação dos rótulos: as classes vêm da
  pasta em que o arquivo está, e o repositório não guarda a anotação de origem
  que as justifique.

---

## Sobre as imagens: privacidade e licença

As 30 radiografias são de **pacientes reais** e estão versionadas neste
repositório. Elas chegaram anonimizadas — nome e identificador substituídos por
UUID, data do estudo zerada, sem instituição nem médico —, e essa anonimização
foi conferida lendo os cabeçalhos: nenhum campo com identificação direta
sobreviveu. **Idade e sexo permanecem**, o que é o padrão em conjuntos de imagem
médica de pesquisa e não constitui identificação direta.

O repositório **não declara a origem do conjunto nem carrega arquivo de
licença**. Quem for reutilizar estas imagens deve checar os termos da fonte
original antes: conjuntos de radiografia de pesquisa costumam permitir uso
acadêmico e restringir redistribuição.

---

## Estrutura

```
main.py                     todo o pipeline: leitura DICOM, histograma, comparação, leave-one-out
metricas.txt                saída versionada da execução — a tabela de resultados acima
Pneumothorax_files.txt      15 caminhos relativos, classe positiva
No_Pneumothorax_files.txt   15 caminhos relativos, classe negativa
requirements.txt            numpy, pydicom, pillow, opencv-python-headless
train/                      as 30 imagens DICOM, uma pasta por classe
```

Um arquivo de código, 30 imagens e duas listas de texto. A simplicidade é o
ponto: o que dá para afirmar sobre o resultado está inteiramente contido no que
dá para ler em cinco minutos.
