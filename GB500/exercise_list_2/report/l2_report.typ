// IMPORTS //


#import "@preview/subpar:0.1.1"
#import "@preview/codly:1.0.0": *


// IMPORTS CONFIGS //

// codly --> Nicer code blocks
#show: codly-init.with()

#codly(
  number-format: none,
  zebra-fill: none,
  radius: 10pt,
  stroke: 2pt + luma(200)
)

// SETTINGS //

#set page(
  paper: "a4",
)


#set heading(
  numbering: "1. ",
)

#set math.equation(
  numbering: "(1)",
)

#set par(justify: true)

#set text(
  size: 12pt,
  font: "New Computer Modern",
  lang: "pt"
)

// SHOW RULES //

#show link: underline

#show heading: set block(spacing: 1.5em)
#show bibliography: set heading(numbering: "1.")
#show par: set block(spacing: 1.5em)


// TITLE PAGES //

#align(center, text(12pt)[
  #image("figures/lncc.png")
  *Laboratório Nacional de Computação Científica \
  Programa de Pós-Graduação em Modelagem Computacional \
  Fundamentos em Redes Neurais e Aprendizagem Estatística
  *
])


#v(7cm)


#align(center, text(14pt)[
  Resolução da Lista de Exercício 2: \
  Multi Layer Perceptron e PCA
])

#v(4cm)

#align(right, text(14pt)[
  João Pedro dos Santos Rocha \
])



#align(bottom + center)[
  Petrópolis-RJ \
  #datetime.today().year()
]

#pagebreak()

#align(center, text(12pt)[
  #image("figures/lncc.png")
  *Laboratório Nacional de Computação Científica \
  Programa de Pós-Graduação em Modelagem Computacional \
  Fundamentos em Redes Neurais e Aprendizagem Estatística
  *
])


#v(7cm)


#align(center, text(14pt)[
  Resolução da Lista de Exercício 2: \
  Multi Layer Perceptron e PCA
])

#v(3cm)

#align(right, text(11pt)[
])

#grid(
  columns: (2fr, 3fr),
    align(start)[
    ],
    align(end, text(11pt)[
      Trabalho  apresentado  como  parte  dos  critérios de
      avaliação da disciplina de Fundamentos em Redes Neurais e
      Aprendizagem estatística.

      Prof: Gilson Giraldi
    ]))

#align(bottom + center)[
  Petrópolis-RJ \
  #datetime.today().year()
]

#pagebreak()

#set page(
  numbering: "1"
)

#show outline.entry: entry => {
  entry
  v(0.1cm, weak: true)
}

#outline(indent: true)

= Informações gerais

Este documento contém as resoluções para os exercícios da segunda lista de
exercícios #footnote[https://www.lncc.br/~gilson/ML-PR2024/lista2-2024.pdf] da
disciplina de Fundamentos de Redes Neurais e Aprendizagem estatística
ministrada no período 2024.3. As implementações podem ser encontradas no
repositório github
#footnote[https://github.com/jpssrocha/ML_Exercises]<gh_link> na pasta
`GB500/exercise_list_2`.


= Instruções de Reprodutibilidade

Para obter os arquivos usados para o projeto basta baixar o repositório github @gh_link
via a interface gráfica ou no terminal com o comando:

```bash

git clone https://github.com/jpssrocha/ML_Exercises

```

As implementações foram feitas na linguagem de programação Python, por isso
fora adotado o conda
#footnote[https://docs.conda.io/projects/conda/en/stable/]<conda_fn> para a
administração dos pacotes e ambiente virtual. Para reproduzir o ambiente basta
após baixar a pasta entrar na mesma via o terminal e gerar o ambiente via arquivo `environment.yml`.

```bash

cd ML_Exercises
conda env create -f environment.yml

```

Com o ambiente preparado basta ativá-lo e acessar a pasta
`GB500/exercise_list_2/notebooks` e acessar os códigos que estão
no formato de _jupyter notebook_.

```bash

conda activate machine_learning
cd GB500/exercise_list_2/notebooks 
jupyter notebook

```

Isso irá ativar o ambiente e lançar a interface do jupyter de onde será possível
abrir e interagir com as implementações.

OBS: Para que as instruções acima funcionem é necessário instalar o git e o conda
(que pode ser instalado via o anaconda).

Foi utilizado o Python 3.12 e as bibliotecas:

- NumPy: Manipulação de dados numéricos e operações matriciais
- Pandas: Manipulação de dados tabulares
- Matplotlib: Visualização dos dados
- SciPy: Funções auxiliares
- Scikit-learn: Utilitários treinamento de modelos
- PyTorch: Definição e treinamento de rede neurais

= Exercícios


== Questão 1 - MLP para problema de classificação

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  stroke: 1pt,
)[
  #align(center)[Enunciado:] 

  Consider a database and a classification problem. Apply leave-one-out
  multi-fold cross-validation, with K = 4, for a MLP model. Use the facilities
  available in libraries for neural network implementation, like Keras, Tensor
  flow, scikit-learn, Matlab, etc.


   (a) Show the graphical representation of the evolution of training and validation stages. \
   (b) Perform a statistical analysis of the performance of the four models applied over the $DD_(t e)$ .\
   (c) Analyze the influence of optimizer hyperparameters
]


Para iniciar o processo foi necessário escolher uma base de dados.

=== Base de dados e seu tratamento

A base escolhida foi o conjunto de dados de câncer de mama de Wisconsin
@breast_cancer, por duas razões: ser amplamente utilizado para testes, sendo
assim relativamente fácil comparar os resultados obtidos com outros resultados
na literatura, e também por ser pequeno e portanto ter um baixo custo para
avaliar a função custo, agilizando assim a exploração de arquiteturas
diferentes. A mesma pode ser obtida via o repositório de conjuntos de dados da
UCI #footnote(link("https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")).

O conjunto é composto por 539 instâncias com 30 medidas da fisiologia de
tumores de mama, classificados entre maligno ou benigno. O mesmo apresenta um
desbalanceamento de classes por isso foram descartados dados da classe com o
maior número de exemplos (benigno), resultando num conjunto de dados com 412
exemplos.

Além disso o conjunto de dados também apresentava uma grande diferença de
escalas entre as características, com alguns atributos tendo a média  da ordem
de $10^(-2)$ e outros na ordem de $10^2$. Por isso foi usado o MinMaxScaler do
scikit learn para escalonar as colunas entre 0 e 1. 

Por fim as etiquetas de benigno e maligno foram codificadas em _one-hot encoding_
com a função get_dummies do pandas, e separadas as variáveis preditoras das
predições, em duas matrizes diferentes.

=== Arquitetura de rede utilizada

Como necessário para os dados a rede foi configurada com 30 neurônios de
entrada e 2 de saída. Como função de ativação foi usada a ReLu (@eq-relu), que
foi escolhida pelas suas propriedades de treinamento rápido e estabilidade
@Oostwal2021, facilitando assim a experimentação.  Na camada de saída, foi
usada a _softmax_ @eq-softmax para ter uma saída que pode ser interpretada como
probabilidade, que era então transformada na classe mais provável para calcular
a acurácia. Por fim, como função custo fora utilizada a entropia cruzada
(@eq-cross-entropy) por apresentar vantagens em relação ao erro quadrático
médio para tarefas de classificação @Kline2005.

$
"ReLu"(x) = max(0, x)
$ <eq-relu>

$
"softmax"(x_i) = e^(-x_i)/(sum_(j=1) e^(-x_j))
$ <eq-softmax>


$
H(y_("pred"), y_("true")) = sum_i y_("true") log(y_("pred"))
$ <eq-cross-entropy>


Foi escolhida uma arquitetura com 2 camadas escondidas com 5 neurônios na
primeira e 6 na segunda. Esta configuração foi escolhida após uma série de
experimentos simples, seguindo o seguinte procedimento: foi feita uma divisão
simples do conjunto de dados entre treinamento e validação, então partindo de
uma camada com um neurônio, a quantidade de neurônios era aumentada e
realizados 3 treinamentos com descendência estocástica de gradiente (500
épocas, taxa de aprendizagem de 0.1, sem uso de _batches_)  e calculada a mediana
da acurácia num conjunto de validação (para aliviar os efeitos estocásticos por
conta da inicialização dos pesos), isso então foi feito até que não fosse
observada diferença nas primeiras casas decimais, assim era adicionada uma
camada e o procedimento repetido, caso a adição da nova camada não trouxesse
melhoria a camada era descartada e o procedimento terminado. 

Os parâmetros por padrão são inicializados no PyTorch com os bias em 0 e para
os pesos é utilizado algoritmo de inicialização de  Kaiming (ou He)
@Kaiming2015.

=== Validação usando K-Fold

Para realizar a validação cruzada foi utilizada a função kfold do scikit learn
para auxiliar na construção dos _folds_. Como requisitado foram construídos 4
_folds_ onde a acurácia foi calculada para cada _fold_. Relembrando a acurácia é
dada por:

$
"acc" = "classificações corretas"/"classificações incorretas"
$


=== (a) Visualizando a evolução das métricas

Durante o treinamento foram guardadas os valores saídos da função de perda e da acurácia. Um
exemplo pode ser visto na @fig-evolution.

#figure(
  image("figures/train_evolution.png"),
  caption: "Exemplo da evolução das métricas de treinamento e acurácia"
) <fig-evolution>

Para o conjunto de validação foi obtida a seguinte matriz de confusão:

#figure(
    image("figures/confusion_mat.png", width: 60%),
  caption: "Exemplo matriz de confusão de validação."
) 


=== (b) Estatísticas das métricas de performance

Com os 4 _folds_ para o conjunto de treinamento foi obtida a acurácia média de 96.15% com 
desvio padrão de 1.29%. Já no conjunto de validação foi obtido a acurácia média de 95.01%
com desvio padrão de 1.56%.

=== (c) Verificando variações no otimizador

Para verificar a sensibilidade do otimizador foram inicialmente utilizados
diferentes valores para a taxa de aprendizagem (1, 0.5, 0.1, 0.05, 0.01, 0.005,
0.001). A evolução da função de perda para as diferentes taxas pode ser vista
na @fig-lr. É possível observar que o algoritmo converge mais rapidamente para
taxas de aprendizagem mais altas porém com a taxa igual a 1 aparecem estruturas
de oscilação na evolução da função de perda. Pelas figuras é razoável afirmar
que a taxa ideal está próxima de 0.5. Também é possível observar que quanto mais 
distante deste valor a função de perda decai mais lentamente, e quando chegando 
próximo de 0.01 o decaimento é tão pequeno que é difícil de visualizar na figura.

#figure(
  image("figures/different_lr.png"),
  caption: "Evolução da função de perda para diferentes taxas de aprendizagem"
) <fig-lr>


#pagebreak()

== Questão 2 - Complexidade do treinamento de uma MLP

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  stroke: 1pt,
)[
  #align(center)[Enunciado:] 

  Give the worst-case computational complexity for the training process of MLP
  model depicted in Figure 9.6 of the monography. Suppose a generic activation
  function and a generic loss function.
]

A arquitetura desejada pode ser vista na @fig-nn-mono.

#figure(
  image("figures/nn_cost.png", width: 75%),
  caption: "Arquitetura desejada. (Fig 9.6 da monografia)"
) <fig-nn-mono>

Dado que foi pedido o custo do pior caso, será assumido tamanho de _batch_ 1, ou
seja, a cada dado avaliado será dado um passo, o que maximiza o número de
operações de atualização dentro de um dado número de épocas. Também pela mesma
razão será assumido que dadas as camadas $[1, 2, ..., L]$, a quantidade de
neurônios das mesmas, formada pela lista $P = (p(l)| l in [1, ..., L])$ serão
todos iguais a $max(P) = S$.

Para facilitar a contagem das operações de básicas o processo será dividido em
partes e combinado ao final para montar o processo de treinamento. Teremos a
fases:

+ _Forward_
+ Derivadas em relação aos neurônios
+ Derivadas em relação aos parâmetros
+ Atualização dos pesos
+ Custo combinado de treinamento

Serão usadas as equações na forma matricial para facilitar a contagem das
operações, mas com uma nomenclatura semelhante a da monografia do curso, as
diferenças são que que, o subscrito será retirado pois não haverá necessidade
indexar elemento por elemento, pois o vetor ou matriz irá representar todos os
elementos numa dada camada e será também omitido o indexação por ponto de
dados, pois a análise será feita para um ponto de dado e então multiplicado
pelos $N$ pontos de treinamento, simplificando assim a notação. Também, como já
mencionado, será adotado $L$ como índice da camada de saída (haverão L-2
camadas escondidas). Para denotar a função de ativação genérica será usado
$sigma$ e a função custo genérica será $cal(L)$. E o custo será denotado por
$CC$.

As matrizes de pesos de cada camada $W^l$ serão  montadas com cada linha
simbolizando os pesos de um neurônio da camada anterior em relação a cada
neurônio da camada $l$.

Vale lembrar que o passo inicial para o processo de treinamento é inicializar
as matrizes de parâmetros, bem como as variáveis auxiliares para guardar os
resultados intermediários. No entanto este custo não será computado nesta
análise pois não será avaliada a latência de acesso à memória ou complexidade
de espaço. As variáveis inicializadas mais importantes e a notação que será
usada para as mesmas são:

- Contador de épocas - $EE$ - Inteiro
- Registro de entradas e saídas - $"Nets"$, $O$ - Listas com $L$ entradas que são vetores
- Registro de deltas - $Delta_("all")$ - Lista com $L$ entradas que são vetores
- Acumulador de gradientes nos pesos - $"List"(nabla cal(L)_W)$ - Lista com $L$ entradas que são matrizes
- Acumulador de gradientes nos bias - $"List"(nabla cal(L)_B)$ - Lista com $L$ entradas que são vetores

Uma implementação seguindo este esquema foi feita e testada e pode ser vista no caderno
`q2_backpropagation.ipynb` do repositório com as implementações. O algoritmo apresentado 
aqui segue o mesmo fluxo.

Para iniciar o processo de treinamento, começamos pelo _forward_ em um ponto do
conjunto de de treinamento, para iniciar o processo de _backpropagation_ sob esse
ponto.

=== _Forward_

Para o _forward_, usando as premissas de pior caso, teremos para cada neurônio
nas camadas escondidas $S$ operações de multiplicação, $S-1$ de soma, uma
operação de soma do bias e o custo da aplicação da função de ativação $sigma$.
Teremos $S$ neurônios por camada logo teremos $S$ vezes este custo. E isso será
aplicado desta forma para cada camada escondida, então se temos $L$ camadas e duas
são as de entrada e saída teremos $L-2$ vezes este custo. Para finalizar temos que
contar o custo da camada de saída, que sera o custo de um único perceptron. 

Durante o _forward_ as entradas e saídas de cada camada são guardadas para uso
posterior ($"Nets"$ e $O$), no entanto como dito anteriormente não serão incluídos na análise
custos de acesso à memória.

Assim, matematicamente temos:

$
CC("forward") &= S dot CC("perceptron") dot (L-2) + CC("perceptron") \
CC("forward") &= (S dot (L-2) + 1) CC("perceptron") 
$ <eq-custo-forward>

Onde:

$
CC("perceptron") &= S + (S-1) + 1 + CC(sigma) \
                 &= 2S + CC(sigma)
$

Onde $CC(sigma)$ não é conhecido pois $sigma$ é uma função de ativação
genérica.

=== Deltas sob neurônios

Para calcular os deltas começamos pela última camada com a expressão @eq-delta-last.

$ Delta^L = cal(L)'(d^L - o^L) dot.circle sigma'("net"^L) $ <eq-delta-last>

Onde "$dot.circle$" indica multiplicação elemento a elemento e $L$ o índice da
camada de saída. Em seguida os deltas das camadas subsequentes serão calculadas
em função das camadas anteriormente calculadas, tendo a última camada como
ponto de partida. 

$ Delta^l = (W^(l+1) Delta^(l+1)) dot.circle sigma'("net"^l) $ <eq-delta-middle>

Onde $W^(l+1)$ é a matriz de peso da camada seguinte (última que foi calculada). Esses
deltas são guardados a cada camada para ser usados na próxima etapa. 

Assim, na camada final já que só temos 1 neurônio teremos, uma operação de
subtração, mais a avaliação da derivada da função custo que não é conhecida,
mais a avaliação da derivada da função de ativação, que não é conhecida, mais uma 
multiplicação. Matematicamente:

$
CC(Delta^L) &= 1 + CC(cal(L)') + CC(sigma') + 1 \
            &= CC(cal(L)') + CC(sigma') + 2
$

Na penúltima camada ($L-1$) temos que $Delta^(l+1)$ = $Delta^L$, sendo assim,
teremos apenas um valor no vetor $Delta^(l+1)$ e uma coluna na matriz $W^(l+1)$
da @eq-delta-middle, pois só há um neurônio de _output_, logo precisamos computar
o custo dessa camada separadamente das outras camadas que tem tamanho igual
entre si. Neste caso o tamanho da matriz  $W^(l+1)$ na @eq-delta-middle é $S x
1$ e do vetor $Delta^(l+1)$ é $1 times 1$ logo multiplicando os dois teremos $S$
operações de multiplicação. O tamanho do vetor $"net"^(L-1)$ será $S times 1$, logo
teremos que avaliar $sigma'$, $S$ vezes e então para multiplicar entrada a
entrada com $W^L Delta^L$  teremos $S$ multiplicações. Matematicamente temos:

$
CC(Delta^(L-1)) &= S + S CC(sigma') + S \
                &= S (2 + CC(sigma'))
$

Para cada camada $l$ tal que $l in [2, L-2]$, ou seja, desde a primeira camada
escondida até a penúltima, teremos camadas de tamanho $S$, logo teremos que
computar o mesmo custo $L-3$ vezes, sendo todas as camadas menos as camadas de
entrada, saída e a penúltima. No domínio mencionado as matrizes de peso
$W^(l+1)$ terão tamanho $S times S$ e o vetor de pesos $Delta^(l+1)$ terá tamanho
$S times 1$. Logo teremos que computar $S$ produtos internos com $S$ multiplicações
e $S - 1$ somas na primeira parte da @eq-delta-middle. O vetor $"net"^l$ terá tamanho $S$
logo teremos que computar $S$ vezes $sigma'$ e fazer $S$ multiplicações. Matematicamente temos:

$
CC(Delta^l) &= S(S + (S - 1)) + S + S CC(sigma') \
            &= S(2S -1) + S + S CC(sigma') \
            &= S(2S + CC(sigma'))  && forall l in [2, L-2]\
$

Logo o custo total para computar todos os deltas é a soma dos deltas da
ultima camada, mais os da penúltima camada, mais o das $L - 3$ camadas
restantes.

$

CC(Delta's) &= CC(Delta^L) + CC(Delta^(L-1)) + (L-3) CC(Delta^l) \
            &= (CC(cal(L)') + CC(sigma') + 2) + S( CC(sigma') + 2) + (L-3)(S(2S + CC(sigma')))
$ <eq-custo-deltas>

O resultado dessas operações são guardados na lista $Delta_("all")$ para uso na
etapa seguinte.

=== Gradiente sob parâmetros

O gradiente sob os parâmetros pode ser calculado matricialmente com a @eq-grad-w e @eq-grad-b.

$
nabla cal(L)_(W^l) = O^(l-1) (Delta^l)^T \
$ <eq-grad-w>
$
nabla cal(L)_(B^l) = Delta^l
$ <eq-grad-b>

Serão computadas as operações para os pesos e bias e  separadamente para
facilitar a organização. Novamente será necessário computar as operações na
ultima camada primeiro e depois o resto. Avaliando a @eq-grad-w na ultima camada a
matriz de _outputs_ ($O^(L-1)$) terá a forma $S times 1$ e os deltas $Delta^(L)$
transpostos serão $1 times 1$. Logo teremos $S$ multiplicações. Depois precisamos
acumular este gradiente somando na posição $L$ da lista de acumuladores $"List"(nabla
cal(L)_W)[L]$, logo são $S$ somas. Assim:
$
CC(nabla cal(L)_(W^L)) = S + S = 2S
$

Nas camadas intermediárias ($l in [2, L-1]$) teremos a matriz $O^(l-1)$ da
forma $S times 1$ e a matriz $(Delta^l)^T$ na forma $1 times S$. Logo a saída terá a
forma $S times S$ e teremos $S$ multiplicações pra formar cada linha, e $S$ linhas.
E para acumular na lista $"List"(nabla cal(L)_W)[l]$ teremos o número de elementos em
operações de soma $S^2$

$
CC(nabla cal(L)_(W^l)) = S S + S^2 = 2S^2 #h(1cm)  forall l in [2, L-1] 
$

Teremos $L-2$ vezes este custo. Logo somando tudo teremos:

$
CC(nabla cal(L)_W) = 2S + (L-2) 2 S^2 = 2S(1+(L-2)S)
$

Para a atualização dos bias, temos que o próprio delta em relação aos outputs é
o gradiente então, uma vez que já foram calculados, só temos o custo de
acumular estes deltas. Assim na camada de saída temos uma soma, e depois
temos S somas para cada uma das $L-2$ camadas escondidas. Logo temos:

$
CC(nabla cal(L)_B) = 1 + S(L-2)
$

Para cada gradiente acumulado é incrementado um contador que contabiliza quantas vezes já
foi acumulado, ou seja, uma soma. Assim o custo combinado fica:

$
CC(nabla) &= CC(nabla cal(L)_W) + CC(nabla cal(L)_B) + 1
          &= (2S(1+(L-2)S) + (1 + S(L-2))  + 1
$ <eq-custo-grad>


=== Atualização dos pesos e bias

As expressões genéricas para atualizar os parâmetros são:

$
W_(n+1)^l = W_(n)^l + eta nabla W_(n)^l
$

$
B_(n+1)^l = B_(n)^l + eta nabla B_(n)^l
$

Onde o subscrito sinaliza o número do passo e $eta$ a taxa de aprendizagem.
Para fazer a atualização dos pesos temos na última camada $S$ multiplicações do
vetor do gradiente dos pesos pela taxa de aprendizagem $S$ divisões pelo
contador de gradientes acumulados, e $S$ somas. Para o bias temos uma
multiplicação, uma divisão e uma soma. Logo:

$
CC("atualização"^L) = S + S + S + 1 + 1 + 1 = 3(S+1)
$

Para as $L-2$ camadas escondidas teremos uma matriz $S times S$ de pesos, logo
$S^2$ multiplicações pela taxa de aprendizagem, $S^2$ divisões pelo contador de
gradientes acumulados e $S^2$ somas da matriz de gradientes acumulados com a
matriz de parâmetros. E para os bias temos $S$ multiplicações, $S$ divisões e $S$ somas. Logo
temos:


$
CC("atualização"^l) = S^2 + S^2 + S^2 + S + S + S = 3(S^2+S)
$

Ao fim de cada atualização os vetores e matrizes de gradientes acumulados e o
contador são zeradas. No total temos:

$
CC("atualização") = 3(S^2 + S)(L-2) + 3(S+1)
$ <eq-custo-att>

Com isso podemos computar o custo para cada etapa. Agora iremos contabilizar
todas as operações necessárias para treinar $N$ pontos de dados, ao longo de
$EE$ épocas.

=== Custo total de treinamento

Iniciando o processo de treinamento precisamos inicializar as variáveis
auxiliares e então para cada época iremos embaralhar os dados, porem este custo
é complexo de computar e não é o foco desta análise, portanto o mesmo será
desconsiderado. Assim seguindo com o algoritmo para cada dado de treinamento
iremos fazer o _forward_, calcular os deltas, calcular os gradientes e dado que
foi assumido um _batch_ de tamanho 1 será feita uma atualização a cada ponto.
Logo temos:

$
CC("treinamento") = EE &N (CC("forward") + CC(Delta) + ...\
                       ... &+ CC(nabla) + CC("atualização"))
$ <eq-treino-inicial>

Para facilitar a visualização dos termos vamos expandir as equações
#ref(<eq-custo-forward>, supplement: none), #ref(<eq-custo-deltas>, supplement: none),
#ref(<eq-custo-grad>, supplement: none) e #ref(<eq-custo-att>, supplement: none),
agrupando os termos constantes e alinhando os termos de mesmo grau. Temos:

$
CC("forward")     &= 2S^2 L &- 4S^2 &+   S L CC(sigma)  &+ 2S     &- CC(sigma) \
CC(Delta)         &= 2S^2 L &- 6S^2 &+   S L CC(sigma') &+ k_1 S  &+ k_2 \
CC(nabla)         &= 2S^2 L &- 4S^2 &+   S L            &         &+ 2 \
CC("atualização") &= 3S^2 L &- 6S^2 &+ 3 S L CC(sigma)  &- 3S     &+ 2 \
$

Onde:

$
k_1 &= 2 - 2CC(sigma) \
k_2 &= CC(cal(L)') + CC(sigma') + 2
$

Somando as 4 equações resultantes para formar o núcleo da  @eq-treino-inicial, temos:

$
9 S^2 L - 20 S^2 + k_3 S L + k_4 S + k_ 5
$

Onde:

$
k_3 &= CC(sigma) + CC(sigma') + 4 \
k_4 &= k_1 - 1 \
k_5 &= k_2 + CC(sigma) + 4
$

Portanto temos que o treinamento tem o custo:


$
CC("treinamento") &= EE dot N ( 9 S^2 L - 20 S^2 + k_3 S L + k_4 S + k_ 5) \
$ <eq-custo-total>


=== Complexidade de pior caso

Queremos a complexidade de pior caso, ou seja, o limite superior para o custo,
portanto queremos o custo em notação $O$. Assim podemos propor que $ CC("treinamento") = O(EE N  S^2 L)$.

#pagebreak()

*Prova*:

Precisamos mostrar que $exists c in RR $ e  $exists EE_0, N_0, S_0, L_0 in NN $ tal que:

$
CC("treinamento") <= c g(EE_0, N_0,  S_0,  L_0) 
$ <eq-bigo>

Onde $g(EE, N,  S,  L) = EE dot N dot S^2 dot L$. Podemos reorganizar a @eq-custo-total tirando
$S^2 L$ em evidência no fator da esquerda.

$
CC("treinamento") = EE N S^2 L (9 - 20/L + k_3/S + k_4/(S L) + k_5/(S^2 L))
$

Nesta forma podemos notar que no limite onde $L, S -> oo$ o termo $(9 - 20/L + k_3/S + k_5/(S^2 L)) -> 9$. 
O que nos deixa com a equação:

$
lim_(S,L->oo) CC("treinamento") = 9 EE N  S^2 L 
$

Logo é assegurado que existe $L_0$ e $S_0$ que satisfazem a inequação
#ref(<eq-bigo>, supplement: none) se $c > 9$ e $EE_0, N_0 >= 1$. Logo existem
$EE_0, N_0,  S_0,  L_0$ que satisfazem a inequação com $c>9$ e $S_0$ e $L_0$
suficientemente grandes, o quão grandes pode ser determinado conhecendo $k_3,
k_4$ e $k_5$.

#pagebreak()

== Questão 3 - PCA para pequenas amostras

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  stroke: 1pt,
)[
  #align(center)[Enunciado:] 

  Study the theory of PCA for small sample size problems, where the number of
  data points is smaller than the data space dimension. Choose an image
  database convert the images to grayscale and apply the theory of ‘PCA for
  small sample size problems’ for dimensionality reduction.

  (a) If $x$ is the sample mean (centroid of the dataset) and $p_1$ is the
  principal component, visualize the result of expression:

  $
    x = accent(x, -) + alpha p_1
  $

  where α ∈ {$−β sqrt(λ)_1$ , 0, $β sqrt(λ)_1$ } with $lambda_1$ being the eigenvalue associated to $p_1$ and $beta$ a scalar factor

  (b) Study the spectrum of the matrix $X^T X$ to perform dimensionality reduction. Visualize some
  images in the space of reduced dimension.

  (c) Constuct an image generator using the $d$ principal components choosen in item b.
]

Antes de iniciar a implementação dos itens será descrita a base de dados e sua
preparação.

=== Base de dados

O conjunto de dados escolhido foi a base de faces da FEI 
#footnote("https://fei.edu.br/~cet/facedatabase.html")<link-fei>@fei_database que é uma coleção de 400
fotos de 200 pessoas, para cada pessoa há uma foto sorridente e uma foto
neutra. Vamos carregá-lo em uma única matriz $X$ que terá todos os dados.

Os dados podem ser encontrados no site do . Os
arquivos específicos foram as versões cortadas e alinhadas automaticamente,
que podem ser acessados pelos links abaixo.

#align(center)[
- #link("https://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part1.zip")[Rostos alinhados e cortados - Parte 1]
- #link("https://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part2.zip")[Rostos alinhados e cortados - Parte 2]
]

Para usá-los com a implementação no caderno `q3_eigenfaces.ipynb`, baixe os
dois links, descompacte em uma pasta e defina a variável `PATH` na célula
apontada no caderno.

=== Processamento dos dados

Nesta base de dados todos os valores já estão normalizados entre $[0, 255]$ e
em escala de cinza como na @fig-example. Logo só foi necessário carregar
e montar a matriz de dados $X$. 

#figure(image("figures/example.png", width: 75%), caption: "Exemplo de imagem da base FEI") <fig-example>

A matrix de dados foi montada por linhas, ou seja, cada imagem foi transformada num vetor linha e concatenadas numa
matriz. Para montar a matriz de dados fora usado o código:

```py

import numpy as np
import matplotlib.pyplot as plt

imgs = []

for img in IMAGE_FILES:
    imgs.append(plt.imread(img).reshape(1, -1))

X = np.hstack(imgs)

```

=== Aplicação do PCA para pequenas amostras

Para iniciar a aplicação do PCA foi calculada a "face média" (@fig-face_media)
para ser removida linha a linha da matriz de dados resultando numa matriz $A$
de dados removidos da média. Em seguida foi calculada a matriz de covariâncias
$A A^T$ seguindo a teoria em @eigenfaces1991 e @tese_laura.

#figure(
  image("figures/face_media.png", width: 75%),
  caption: "Face média calculada dos dados"
)<fig-face_media>

Para os passos iniciais fora usado:

```py

X_bar = np.mean(X, axis=0)  # Calculando face média

A = X - X_bar[:, np.newaxis]  # Criando matriz A de dados centralizados

AAt = A@A.T/(N-1)  # Calculando a matriz de covariâncias

```

Em seguida foi computado os autovalores e autovetores desta matriz usando a
função `np.linalg.eigh` que calcula autovetores de forma otimizada para
matrizes hermitianas. Esta função entrega os autovalores e autovetores
associados em ordem crescente, por isso para o PCA é necessário inverter a
ordem para ter os valores na ordem decrescente. Com isso podemos montar a 
matriz de projeção usando a teoria de PCA para amostras pequenas:

$
P_("PCA") = (A^T V)/(||A^T V||_(2 "cols"))
$

Onde $V$ é a matriz de autovetores de $A A^T$ montada por colunas em ordem
decrescente de valor do autovetor associado, e a divisão do vetor de normas por
coluna indica uma normalização da matriz $A^T V$ por colunas. Isso é feito
usando o código abaixo.

#pagebreak()

```py

eigvals, eigvec =  np.linalg.eigh(AAt)
eigvals = eigvals[-1::-1]
eigvecs = eigvec[:, -1::-1]

P_pca = A.T@eigvecs
P_pca = P_pca/np.linalg.norm(P_pca, axis=0)

```

Com a matriz $P_("PCA")$ calculada podemos iniciar a resolução dos itens do enunciado.

=== (a) "Face principal" 1

Para visualizar o resultado de:

$
  x = accent(x, -) + alpha p_1
$

Com α ∈ {$−β sqrt(λ)_1$ , 0, $β sqrt(λ)_1$ }, escolhendo $beta = 1.5$, foi usado o código:

```py

PC1 = (
    (1.5*np.sqrt(eigvals[eig_n])*P_pca[:,eig_n] + X_bar)
    .reshape(example.shape)  # Reformatando vetor como imagem
)

plt.imshow(PC1, cmap="gray")

```

Tendo como resultado a @fig-pc1. Que apresenta uma expressão neutra, o que pode
ser explicado olhando a @fig-pc1_pure que mostra a $p c_1$ sem adicionar a face
média, na escala de cores é possível visualizar que a esta componente
correlaciona (valores positivos) com estruturas que parecem com as rugas de
expressão que geralmente se sobressaem numa expressão neutra, enquanto
anti-correlaciona (valores negativos) com estruturas parecidas com dentes, que
normalmente iriam se sobressair numa pessoa com a expressão sorridente.

#subpar.grid(
  figure(image("figures/pc1.png"),
  caption: $accent(x, -) + alpha p_1$), <fig-pc1>,
  figure(image("figures/pure_pc1.png"),
  caption: $p_1$), <fig-pc1_pure>,
  columns: (1fr, 1fr),
  caption: "Visualizações da PC1",
  inset: 0pt,
  column-gutter: -30pt
)

Repetindo o mesmo exercício com a $p c_2$ é possível visualizar que a mesma
apresenta uma face sorridente. Olhando sem a adição da face média é possível
ver que a mesma correlaciona bastante com as áreas ao redor da boca e ao redor
dos olhos que normalmente ficam com vários detalhes ao sorrir.

#subpar.grid(
  figure(image("figures/pc2.png"),
  caption: $accent(x, -) + alpha p_2$), <fig-pc2>,
  figure(image("figures/pure_pc2.png"),
  caption: $p_2$), <fig-pc2_pure>,
  columns: (1fr, 1fr),
  caption: "Visualizações da PC2",
  inset: 0pt,
  column-gutter: -30pt
)


=== (b) Visualizando faces no espaço reduzido

Usando a matriz $P_("pca")$ para projetar as imagens no espaço reduzido nas
PC's 1, 2 e 3 temos a figura @fig-projection. Nesta figura foram marcados as
projeções dos pontos de rostos sorridentes e com expressões neutras.
Aparentemente a dimensão de maior variância fora a que distinguia faces
sorridentes de neutras, o que é bastante plausível dado que a base é dividida
metade a metade entre imagens sorridentes e neutras.

#figure(
  image("figures/pc123.png", width: 70%),
  caption: "Projeção nas PC's 1, 2 e 3 do grupo de imagens sorridentes vs neutras"
) <fig-projection>


Para calcular a variância explicada das PC's foi usado o código abaixo.

```py
cum_explained_var = np.cumsum(eigvals)/ np.sum(eigvals)*100
```

Plotando num gráfico (@fig-cumvar) permitiu identificar que com aproximadamente
50 PC's é possível recuperar 80% da variância nos dados, com 90 é possível
recuperar 90%, com 150 é possível recuperar 95% e com 300 é possível recuperar
99%.

#figure(
  image("figures/var_explained.png", height: 30%),
  caption: "Percentual acumulativo da variância total"
) <fig-cumvar>

Podemos visualizar a qualidade visual desses cortes usando a matriz $P_("pca")^T$ 
para reprojetar os pontos no espaço original. Na @fig-rec podemos visualizar o contraste
entre as reconstruções usando diferentes quantidades de componentes.

#subpar.grid(
  figure(image("figures/example.png"),
  caption: "Original"), 
  figure(image("figures/pca_reconstructions.png"),
  caption: "Reconstruções"), <fig-pc_mosaic>,
  columns: (1fr, 1fr),
  caption: "Figura Original vs Reconstruções",
  inset: 0pt,
  column-gutter: -30pt,
  label: <fig-rec>
)


=== (c) Gerador de faces

Para a geração de bases fora inicialmente usada uma distribuição normal n-dimensional, tendo
como média o vetor $arrow(0)$ e como matriz de covariância, a matriz de
autovalores do PCA divida por 3/4 para diminuir a possibilidade dos pontos gerados
fugirem da distribuição original.

```py

np.seed.random(123)
random_vals = np.random.multivariate_normal(
                  np.zeros(eigvals.shape[0]),
                  eigvals_mat, size=100
)

```

Amostrando esta distribuição foram gerados pontos que seguem um padrão similar
ao dos dados, no entanto, é possível perceber nas PC's 1 e 2 que há uma
multimodalidade que faz com que os pontos gerado sigam um padrão que não
acompanhe a distribuição neste plano como pode ser visto na @fig-random_planes.
Com esses pontos aleatórios foi observado que as reconstruções ficavam com a
aparência da boca bastante distorcida, pois misturava provavelmente misturava
informação de PC's relacionadas a sorriso aos de expressão neutra.

#subpar.grid(
  figure(image("figures/random_pc12.png"),
  caption: "PC1 vs PC2"), <fig-random_pc12>,
  figure(image("figures/random_pc23.png"),
  caption: "PC2 vs PC3"), <fig-random_pc23>,
  columns: (1fr, 1fr),
  caption: "Diferentes planos, com pontos gerados sobrepostos",
  inset: 0pt,
  column-gutter: -30pt,
  label: <fig-random_planes>
)

Para refinar este detalhe foi usando um ajuste simples de mistura gaussiana  de
duas gaussianas multidimensionais nas PCs. Assim foram amostrados pontos da distribuição
resultante. O resultado pode ser visto na @fig-gmm.

```py
from sklearn.mixture import GaussianMixture
np.random.seed(13)

GM = GaussianMixture(2)
GM.fit(A_transform)

gm_sample, _ = GM.sample(100)
```

#figure(
  image("figures/sample_gmm.png", width: 75%),
  caption: "Amostra da Mistura Gaussiana"
) <fig-gmm>


A amostra da mistura gaussiana foi então usada no lugar das PC's geradas
pela gaussiana multidimensional inicial. Com isso foram geradas 100 pontos
aleatórios no espaço das PC's e desses pontos foram usadas as 50 primeiras PC's
(removendo 20% da variância) para reconstruir as faces aleatórias (como na
@fig-random_faces).

#figure(
  image("figures/generated_faces.png"),
  caption: "Exemplos de faces aleatórias"
) <fig-random_faces>

Dentre as faces geradas, muitas ainda tenderam a ter defeitos ao redor da boca
o que pode indicar que a maioria das PC's tenham capturado a variação entre
sorrisos, como pode ser visto na @fig-eigenfaces. Porém os resultados obtidos
foram muito melhores com este refinamento, pois na @fig-random_faces é possível
observar algumas faces convincentes como a 4, 6, e 7, bem como outras, o que
não aconteceu sem este refinamento.

#figure(
  image("figures/eigenfaces.png", width: 80%),
  caption: "Exemplos de autofaces"
) <fig-eigenfaces>

Talvez uma forma de melhorar o processo de geração seria criar modelos
especialistas, ou seja, um com os dados de faces neutras e outro com as faces
sorridentes, assim os padrões marcantes de cada uma não iriam interferir com a
outra, além de diminuir multimodalidades nas PC's.

#pagebreak()
#bibliography("refs.bib")
