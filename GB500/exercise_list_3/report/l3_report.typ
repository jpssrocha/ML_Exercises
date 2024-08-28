#import "lncc_report_template.typ": template

#show: doc => template(
  title: [Resolução da lista final \ SVM, CNN, KPCA, DPCA e LDA],
  author: "João Pedro dos S. Rocha",
  discipline: "Fundamentos em Redes Neurais e Aprendizagem Estatística",
  professor: "Gilson Giraldi",
  bibliography_file: "refs.bib",
  doc
)

= Informações gerais

Este documento contém as resoluções para os exercícios da lista final de
exercícios da disciplina de Fundamentos em Redes Neurais e Aprendizagem
estatística ministrada no período 2024.3. As implementações podem ser
encontradas no repositório github
#footnote[https://github.com/jpssrocha/ML_Exercises]<gh_link> na pasta
`GB500/exercise_list_3`.

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
`GB500/exercise_list_3/notebooks` e acessar os códigos que estão
no formato de _jupyter notebook_.

```bash
conda activate machine_learning
cd GB500/exercise_list_3/notebooks 
jupyter notebook
```

Isso irá ativar o ambiente e lançar a interface do jupyter de onde será
possível abrir e interagir com as implementações. OBS: Para que as instruções
acima funcionem é necessário instalar o git e o conda (que pode ser instalado
via o anaconda).

Foi utilizado o Python 3.11 e as bibliotecas:

- NumPy: Manipulação de dados numéricos e operações matriciais
- Pandas: Manipulação de dados tabulares
- Matplotlib: Visualização dos dados
- Seaborn: KDE plot
- SciPy: Funções auxiliares
- Scikit-learn: Utilitários treinamento de modelos
- PyTorch: Definição e treinamento de rede neurais

#pagebreak(weak: true)

= Exercícios

== Questão 1

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  stroke: 1pt,
)[
  #align(center)[Enunciado:] 

  Consider an image database and a classification problem. Apply leave-one-out
  multi-fold cross- validation explained in section 8.5 of the monography, with
  K = 4, and SVM as follows:

  (a) Non-separable Linear SVM with feature space obtained through the KPCA

  (b) Non-separable Kernel SVM with feature space obtained through the PCA

  (c) Compare the results of items (a) and (b)
]

=== Base de dados

O conjunto de dados escolhido foi a base de faces da FEI
#footnote("https://fei.edu.br/~cet/facedatabase.html")<link-fei>@fei_database
que é uma coleção de 400 fotos de 200 pessoas, para cada pessoa há uma foto
sorridente e uma foto neutra. Foi usado a base normalizada e cortada, que podem
ser acessados pelos links abaixo (assim como na questão 3 da lista anterior).

#align(center)[
- #link("https://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part1.zip")[Rostos alinhados e cortados - Parte 1]
- #link("https://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part2.zip")[Rostos alinhados e cortados - Parte 2]
]

Será considerado o problema de classificação entre face sorridente e neutra.

=== Processamento dos dados

Os dados foram operados da mesma forma que na questão 3 da lista 2. Ou seja,
foram carregados um a um, transformados em vetores e montados numa matriz por
colunas. Novamente os valores já estão normalizados entre 0 e 255. Salvo que
desta vez não foi removida a média pois para o KPCA isso é feito diretamente
sob a matriz de covariâncias.

Como mencionado anteriormente os dados estão divididos em metade sorridente e
metade neutro, ou seja, a base já está balanceada. Logo não foi necessário usar
nenhum tipo de balanceamento para a tarefa de classificação.

=== (a) SVM linear com KPCA

Para este item, bem como os próximos foram usadas as facilidades do scikit
learn, dado que o mesmo possui módulos prontos para o caso não separável da SVM
(com _kernel_ a escolher), bem como para o KPCA 
#footnote[Fora testado no caderno `q1.1_supplement_kpca_by_hand.ipynb`].


A SVM foi utilizada através da classe `SVC`  com os parâmetros padrões do
sklearn e com a opção `kernel="linear"` para que não fosse utilizado
nenhum _kernel_. O KPCA foi utizado através da classe `KernelPCA` configurado
para utilizar o _kernel_ cosseno, com 150 componentes, e para o restante dos
hiper parâmetros foram mantidas as configurações padrão do sklearn. O _kernel_ foi
escolhido testando dentre todos os disponíveis e escolhendo aquele que resultasse na
maior acurácia num conjunto de teste, e o número de PC's foi escolhido com base no
comportamento observado para as PC's no caso linear escolhendo o número de PC's
necessárias pra recuperar 95% da variância.

Para ajustar e rodar ambos SVM e KPCA, em sequência fora usado o utilitário
`pipeline` também do scikit learn. Resultando no modelo definido no código 
abaixo.


```py
model = make_pipeline(
                KernelPCA(n_components=150, kernel="cosine"),
                SVC(kernel="linear"))
```

Assim foram criados 4 modelos como o descrito acima e foi aplicada validação cruzada
com K-fold (K=4), para cada _fold_ foi calculado a acurácia sob o _fold_ de teste, e 
os valores resultantes coletados para análise posterior. Isso foi feito com o código:

```py
# Starting auxiliar variables
kf = KFold(k, random_state=42, shuffle=True)
models = []
accuracies = []

# Training on each fold
for train_i, test_i in kf.split(X, Y):
    models.append(
        model_factory().fit(X[train_i], Y[train_i])
    )
    accuracies.append(models[-1].score(X[test_i], Y[test_i]))
```

Para cada _fold_ de teste ($N=100$) foram obtidas as matrizes de confusão da
@fig-conf-1a. Realizando as estatísticas sob as acurácias nos _folds_ foi
obtida uma média $mu_a = 94.25%$ e desvio padrão $sigma_a = 2.86%$.

#figure(
  image("figures/q1_a.png", width: 65%),
  caption: "Matrizes de confusão para cada fold"
) <fig-conf-1a>



=== (b) SVM kernel com PCA

Para este item foi usada a mesma estrutura anterior mudando apenas o modelo a
SVM fora configurada para usar um _kernel_ de função radial de base e o KPCA
foi configurado para usar o _kernel_ "linear", que se trata do PCA padrão. O
_kernel_ para a SVM foi escolhido pelo mesmo processo anterior, ou seja, foi
feito uma separação simples de treino e validação nos dados e então foram testados
todos os _kernels_ disponíveis e então escolhido o que resultasse na maior acurácia
sob o conjunto de teste.

Em seguida foi feita a validação cruzada usando exatamente o mesmo processo do
item anterior. Obtendo as matrizes de confusão da @fig-conf-1b.

#figure(
  image("figures/q1_b.png", width: 65%),
  caption: "Matrizes de confusão para cada fold"
) <fig-conf-1b>

Foi obtida uma acurácia média $mu_b = 96.25%$ e o desvio padrão $sigma_b = 1.92%$.

=== (c) Comparação dos métodos

As estatísticas para a acurácia nos _folds_ no item (a) foram  $mu_a = 94.25%$
e $sigma_a = 2.86%$ e para o item (b) foram $mu_b = 96.25%$ e $sigma_b =
1.92%$. Aparentemente a configuração (b) foi mais eficiente, no entanto há
bastante sobreposição entre as métricas, como pode ser observado no plot de KDE
da @fig-kde-1c. Logo se torna desejável uma análise mais refinada para
entender se há uma diferença significativa entre os métodos.


Por ser simples porém sistemático, foi escolhido um teste estatístico para
avaliar se há diferença significativa entre as médias das duas distribuições de
_folds_ assumindo independência nas amostras. Dado que foram poucos _folds_ (8
no total, ou 4 para cada configuração) optou-se por usar um teste de permutação
por ser um teste estatístico robusto, não paramétrico e que funciona com
quantidades menores de dados (neste caso: 8! = 40320 permutações) (como
explicado em @Downey2014). Assim foi aplicado o teste descrito em @Downey2014
(sessão 9.3) usando como estatística de teste a diferença entre as médias sob a
hipótese nula de que ambas são iguais, ou seja, $mu_a - mu_b = 0$ e nível de
significância $alpha = 0.05$.

#figure(
  image("figures/q1c_kde.png", width: 75%),
  caption: "Plot KDE para as medidas de acurácia dos folds"
) <fig-kde-1c>


A métrica de teste observada foi $mu_a - mu_b = 0.02$. Nesse procedimento a
distribuição da métrica de teste é obtida via simulações de Monte Carlo. Assim
os dados de ambas as populações são combinados (modelando a hipótese nula), e
então são feitas N permutações (foi usado $N=2 dot 8!$), que são então
divididos aleatóriamente em dois grupos, os quais são usados para calcular a
métrica de teste (diferença entre as médias) gerando assim N valores
diferentes, por fim para calcular o valor-p é calculada a fração dos dados que
é igual ou mais extrema que a métrica de teste observada para os dados reais,
dividido por N. Assim a distribuição da estatística de teste obtida pode ser
visualizada na @fig-1c-test.


#figure(
  image("figures/q1c_test.png", width: 75%),
  caption: "Distribuição da métrica de teste"
) <fig-1c-test>


Foi obtido um valor-p de 0.39, logo com o nível de significância de 0.05 este
teste falha em rejeitar a hipótese nula de que não há diferença nas médias. Ou
seja, não é possível afirmar que há diferença entre as médias com os dados fornecidos
utilizando este teste.

Para estimar se a potência estatística do teste acima é satisfatória foram
feitos experimentos via simulação de monte-carlo para estima-la sob as codições
observadas assumindo uma distribuição normal (i. e. usando dados simulados de
duas gaussianas com a mesma média e desvio padrão observados, e portanto $mu_a - mu_b != 0$), eles se
encontram no caderno `q1.2_supplement_power.ipynb`. 

O procedimento foi uma variação do procedimento descrito em @Landau2013 (sessão
2). Primeiro foram gerados 4 pontos de cada distribuição, o teste foi então
computado e foi verificado se o valor-p resultante era maior ou menor que o
nível de significância, assim os resultados foram guardados como 1 se menor ou
igual, e 0 se maior, assim a média da lista de resultados resulta na proporção
de resultados positivos, ou seja, a proporção de vezes que a hipótese nula foi
rejeitada dado q a mesma (por construção) era falsa.

Assim o valor estimado para a potência estatística foi de aproximadamente 0.15,
ou seja, dada uma situação semelhante à observada nesta análise, assumindo a
hipótese nula é falsa,  então num nível de significância de 5% o teste rejeita
a hipótese nula apenas 15% das vezes. O procedimento foi repetido também para o
teste de Mann-Whitney @MannWhitney1947 (disponível através da biblioteca scipy como
`stats.mannwhitneyu`) que também é um teste não paramétrico para duas amostras
independentes porém a hipotese nula é de que os dados vem da mesma
distribuição. Para este teste a potência estimada foi ainda menor sendo
aproximadamente 11%.

Vale ressaltar que foi assumida uma distribuição normal para os experimentos de
monte-carlo por ser um caso "fácil" em relação à definição da média, já que tem
pouco peso nas caldas, logo os experimentos com ela definem um limite superior
para a potência estatística. Com distribuições arbitrárias a potência 
estimada podeira ser ainda menor.

Ou seja, segundo estes testes não há evidência suficiente com K=4 para ter
confiança que o esquema do item (b) é mais acurado que o esquema (a). Também
foi estimado através dos experimentos que, dada a diferença de médias
observadas seriam necessários pelo menos 45 _folds_ em cada configuração para
conseguir uma potência de pelo menos 95% usando ambos os testes estatísticos
utilizados. Para esta estimativo foi utilizado o mesmo procedimento usado para as 
4 amostras de cada distribuição, porém incrementando o número de _folds_ até passar
de 95% de potência.

#pagebreak(weak: true)
== Questão 2

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  stroke: 1pt,
)[
  #align(center)[Enunciado:] 

  Consider a database and a classification problem. Apply leave-one-out
  multi-fold cross-validation ex- plained in section 8.5 of [2], with K = 5,
  for a CNN model. Use the facilities available in libraries for neural network
  implementation, like Keras, Tensor flow, etc. [3].

  (a) Show the graphical representation of the evolution of training and
  validation stages (see Figure 8.8 of the course monograph). 

  (b) Perform a statistical analysis of the performance (section 8.6) of the
  five models applied over the $DD_(t e)$ .
]

#pagebreak(weak: true)
== Questão 3

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  stroke: 1pt,
)[
  #align(center)[Enunciado:] 

  Consider an image database and a classification problem.

  (a) Apply leave-one-out multi-fold cross-validation explained in section 8.5
  of [2], with K = 4 with LDA in the reduced PCA space and perform
  classification over the test set. Analyze the results.

  (b) Apply leave-one-out multi-fold cross-validation explained in section 8.5
  of [2], with K = 4, and non-separable Kernel SVM with feature space obtained
  through the Discriminant Principal Component Analysis (DPCA).

  (c) Compare the results obtained in items (a)-(b) above.
]
