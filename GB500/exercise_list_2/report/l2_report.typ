#set page(
  paper: "a4",
)

#import "@preview/codly:1.0.0": *
#show: codly-init.with()

#codly(
  number-format: none,
  zebra-fill: none,
  radius: 10pt,
)

#show link: underline

#show heading: set block(spacing: 1.5em)
#show par: set block(spacing: 1.5em)

#set heading(
  numbering: "1."
)

#set par(justify: true)

#set text(
  size: 12pt,
  font: "Arial",
  lang: "pt"
)

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
      avaliação da disciplina de Fundamentos de Redes Neurais e
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

#pagebreak()

= Informações gerais

Este documento contém as resoluções para os exercícios da
segunda lista de exercícios da disciplina de Fundamentos de
Redes Neurais e Aprendizagem estatística ministrada no
período 2024.3. As implementações podem ser encontradas no
repositório github #footnote[https://github.com/jpssrocha/ML_Exercises]<gh_link>
na pasta `GB500/exercise_list_2`.


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

Para iniciar o processo precisamos selecionar uma base de dados. 

=== Base de dados


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

  where α ∈ {$−β λ_1$ , 0, $β λ_1$ } with $lambda_1$ being the eigenvalue associated to $p_1$ and $beta$ a scalar factor

  (b) Study the spectrum of the matrix $X^T X$ to perform dimensionality reduction. Visualize some
  images in the space of reduced dimension.

  (c) Constuct an image generator using the $d$ principal components choosen in item b.
]


= Referências

