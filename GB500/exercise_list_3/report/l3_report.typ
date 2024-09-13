#import "lncc_report_template.typ": template
#import "@preview/subpar:0.1.1"

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
via o anaconda). Foi utilizado o Python 3.11 #footnote([Foi utilizada a versão
  3.11 invés da 3.12, pois a funcionalidade `torch.compile`, que pode aumentar
  em 2x a velocidade de execução, ainda não é suportada na 3.12]) e as
bibliotecas:

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
ser acessados pelos links abaixo.

#align(center)[
- #link("https://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part1.zip")[Rostos alinhados e cortados - Parte 1]
- #link("https://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part2.zip")[Rostos alinhados e cortados - Parte 2]
]

Será considerado o problema de classificação entre face sorridente e neutra.

#subpar.grid(
  figure(image("figures/100a.jpg", width: 50%), caption: "Face neutra"),
  figure(image("figures/100b.jpg", width: 50%), caption: "Face sorrindo"),
  columns: (1fr, 1fr),
  column-gutter: -30pt
)



=== Processamento dos dados

Os dados foram operados da mesma forma que na questão 3 da lista 2. Ou seja,
foram carregados um a um, transformados em vetores e montados numa matriz por
colunas. Novamente os valores já estão padronizados entre 0 e 255. Desta vez
não foi removida a média pois para o KPCA isso é feito diretamente sob a matriz
de covariâncias. Em todo caso já que serão usadas as facilidades do sklearn
estas operações são feitas automaticamente.

Como mencionado anteriormente os dados estão divididos em metade sorridente e
metade neutro, ou seja, a base já está balanceada. Logo não foi necessário usar
nenhum tipo de balanceamento para a tarefa de classificação.

=== (a) SVM linear com KPCA

Para este item, bem como os próximos foram usadas as facilidades da biblioteca scikit
learn, dado que a mesma possui módulos prontos para o caso não separável da SVM 
#footnote[Como a documentação fala em https://scikit-learn.org/stable/modules/svm.html#svc]
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
necessárias pra recuperar 95% da variância (150 PC's). Foi usado o mesmo número
de PC's tanto no caso linear quanto no caso não linear para ter uma comparação 
mais direta.

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
  image("figures/q1_a.png", width: 85%),
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
item anterior. Obtendo as matrizes de confusão da @fig-conf-1b. Foi obtida uma
acurácia média $mu_b = 96.25%$ e o desvio padrão $sigma_b = 1.92%$.

#figure(
  image("figures/q1_b.png", width: 85%),
  caption: "Matrizes de confusão para cada fold",
  placement: auto
) <fig-conf-1b>


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
  multi-fold cross-validation ex- plained in section 8.5 of the course
  monograph, with K = 5, for a CNN model. Use the facilities available in
  libraries for neural network implementation, like Keras, Tensor flow, etc.

  (a) Show the graphical representation of the evolution of training and
  validation stages (see Figure 8.8 of the course monograph). 

  (b) Perform a statistical analysis of the performance (section 8.6) of the
  five models applied over the $DD_(t e)$ .
]

=== Base de dados

Dado que redes neurais convolucionais possuem um grande quantidade de
parâmetros, foi escolhida para esta questão uma base maior, especificamente a
base de dados clássica MNIST #footnote[https://yann.lecun.com/exdb/mnist/], a
versão utilizada é composta por 70 mil imagens de dígitos escritos à mão, de
$28 times 28$  _pixels_ com as devidas etiquetas (sendo 60 mil de treino e 10
mil de validação). As imagens da base já estão normalizadas entre 0 e 1, e em
preto e branco, não necessitando pré-processamento da imagens para a tarefa de
classificação. As etiquetas tiveram que ser transformadas usando um esquema de
_one-hot encoding_. As imagens foram carregadas e agrupadas em _batches_ usando
as facilidades da biblioteca PyTorch.

#figure(
  image("figures/mnist_example.png", width: 65%),
  caption: "Exemplo de imagem do mnist"
) <fig-mnist-ex>

Foi verificado o balanço do conjunto de dados, e observado que o mesmo não é
perfeitamente balanceado, no entanto está bem próximo, com todas as classes
tendo aproximadamente 6 mil ocorrências de treino e mil de validação. Na
@fig-class-balance é possível ver a distribuição entre as classes no conjunto
de treinamento. Portanto não foi aplicada nenhuma técnica de balanceamento de
classes.

#figure(
  image("figures/class_balance.png", width: 50%),
  caption: "Histograma das classes no conjunto de treinamento"
) <fig-class-balance>

=== Arquitetura da rede

A rede fora também definida utilizando a biblioteca PyTorch. A arquitetura da
rede foi escolhida via tentativa e erro partindo de arquiteturas comumente
utilizadas para classificação de imagens pequenas #footnote[Estas arquiteturas
  foram observadas dentre as analises compartilhadas no site www.kaggle.com],
esta aproximação foi escolhida pois o tempo de processamento para treinar no
computador utilizado (10-20 min) tornou inviável esperar o tempo necessário
para testar muitas combinações. Foram feitos alguns experimentos em cima dos
dados de teste e treinamento até obter uma acurácia acima de 0.95 no conjunto
de teste, usando descendência de gradiente estocástica com taxa de aprendizagem
de 0.5 tamanho de _batch_ de 50 e com 4 épocas #footnote[a partir de 5 épocas
  estava sendo observado _overfitting_].

Desta forma foi escolhida uma arquitetura iniciando por um bloco de 16
convoluções com _kernels_ $ 7 times 7$, seguidos por uma camada de _max pooling_,
então os dados eram vetorizados e passados para uma rede neural densa com
1936 neurônios na primeira camada ($16 times 11 times 11$) seguida uma função
de ativação _ReLU_, então uma camada de saída com 10 neurônios que então eram
passados para uma função de ativação _softmax_. 

Esta configuração ultrapassou a acurácia desejada, chegando à 98% nos testes
feitos para definir a arquitetura da rede. Na @fig-confusion-ex é possível ver
a matriz de confusão sob as 10 mil amostras de teste.

#figure(
  image("figures/example_confusion.png", width: 75%),
  caption: "Exemplo de matriz de confusão obtida nos testes iniciais",
  placement: auto,
) <fig-confusion-ex>


 Aparentemente esta arquitetura encorajou a rede a aprender
filtros de convolução que destacam as bordas em diferentes direções, como pode
ser observado na @fig-conv-ex.

#figure(
  image("figures/convolution.png", width: 50%),
  caption: "Exemplo de saída do bloco de 16 convoluções aprendidas",
) <fig-conv-ex>


=== (a) Evolução do treinamento

Durante os treinamentos foram guardados os valores para a função de perda e
acurácia sob os batches de treinamento e de validação. Um exemplo da evolução
do treinamento pode ser visto abaixo na @fig-evolution, para uma das rodadas de experimentação. 


#figure(
  image("figures/example_evolution.png", width: 90%),
  caption: "Evolução das métricas de loss e acurácia durante o treinamento, na experimentação",
  //placement: auto,
) <fig-evolution>

Para verificar se os resultados finais obtidos são robustos, foi então feita a
validação cruzada.

=== (b) Estatísticas da validação cruzada (K=5)

Nesta etapa foi usado novamente as facilidades do PyTorch combinadas com o
objeto KFold do sklearn  para fazer a divisão dos folds e treinamento dos k
modelos (k=5). Foram obtidos os 5 valores de acurácia sob a partição de
validação obtendo a acurácia média de 97.89% com desvio padrão de 0.26%.

Na @fig-confusion-k1 podemos ver a matriz de confusão do fold 1, é possível
observar que a acurácia das classificações ficou equilibrada no geral, apenas
tendo uma tendência um pouco maior de imagens do dígito 9 serem classificadas
de forma errada, que pode ser visto na linha do dígito 9. Como esperado temos
uma maior quantidade de dígitos 5 devidamente classificados, isso se dá pois há
uma quantidade um pouco maior de dados para o mesmo no conjunto de dados.


#figure(
  image("figures/cnn_confusion_k=1.png", width: 75%),
  caption: "Matriz de confusão com k=1",
) <fig-confusion-k1>

Para ter uma visão geral das 5 matrizes de confusão as mesmas foram empilhadas
numa terceira dimensão, formando um tensor e foi tirado a média e desvio padrão
ao longo desta dimensão, os mesmos podem ser vistos na @fig-confusion-mean e
@fig-confusion-std.

#subpar.grid(
  figure(
    image("figures/q2_mean_conf.png"),
    caption: "Matriz de confusão média",
  ), <fig-confusion-mean>,

  figure(
    image("figures/q2_std_conf.png"),
    caption: "Matriz de confusão de desvios padrão",
  ), <fig-confusion-std>,
  columns: (1fr, 1fr),
  column-gutter: -30pt,
  caption: "Matrizes de confusão agregadas"
)

A tendência a errar a classificação do dígito 9 foi diminuída ao longo dos
folds, portanto provavelmente aquele viés era específico do modelo k=1. A grade
de desvios padrão foi dividida pela grade de médias (somado com 1 para evitar
divisões por zero) para facilitar a visualização das células sem o efeito de
escala. O resultado pode ser visto na @fig-confusion-norm.

#figure(
  image("figures/q2_std_norm_conf.png", width: 75%),
  caption: "Matriz de confusão de desvios padrão normalizada pela média",
) <fig-confusion-norm>

Podemos observar uma maior variabilidade nos digitos 1 sendo classificados como
2 e dos dígitos 5 sendo classificados como 8. Também é fácil ver que os
elementos na diagonal foram os que menos variaram em relação à escala. No geral
foi observado pouca variação entre folds.

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
  of the monography, with K = 4 with LDA in the reduced PCA space and perform
  classification over the test set. Analyze the results.

  (b) Apply leave-one-out multi-fold cross-validation explained in section 8.5
  of the monography, with K = 4, and non-separable Kernel SVM with feature space obtained
  through the Discriminant Principal Component Analysis (DPCA).

  (c) Compare the results obtained in items (a)-(b) above.
]

=== Base de dados e processamento

Para este exercício será novamente usada a base de dados de faces da FEI, como na questão 1
usando os mesmos passos de processamento.

=== (a) Classficação usando LDA

Neste item foi combinado a projeção na direção do LDA, através do objeto
`LinearDiscriminantAnalysis`, com uma classificação usando uma SVM, através do
objeto `LinearSVC`, ambos do scikit learn. Pra isso foi montada uma pipeline como 
mostrado no código abaixo.

```py
make_pipeline(
    LinearDiscriminantAnalysis(),
    LinearSVC(),
) 
```

Na @fig-lda-proj podemos ver um exemplo da projeção dos dados na direção dada
pelo LDA. Que foram então passados para a SVM para definir o ponto de
separação.

#figure(
  image("figures/q3_a_proj_lda.png", width: 50%),
  caption: "Projeção de todos os dados na direção dada pela LDA",

) <fig-lda-proj>

Após a definição do modelo, foi feito o treinamento e validação cruzada do
classificador via k-fold com k=4. Foram obtidas as matrizes de confusão
apresentadas na @fig-conf-3a, através delas foi possível ver que o
classificador se mostrou bastante acurado e ao errar não apresentou um viés 
para falso positivo, ou falso negativo.

#figure(
  image("figures/q3_a_cross_val.png"),
  caption: "Matrizes de confusão da validação cruzada"
) <fig-conf-3a>

No geral foi obtido uma acurácia de 96.75% com desvio padrão de 1.48%.
Apresentando um bom desempenho global.

=== (b) Classficação usando DPCA

Para este item, dado que o DPCA não é um método padrão do sklearn foi 
desenvolvida uma pequena classe para fazer o mesmo funcionar devidamente
com as facilidades do sklearn. A mesma pode ser vista no código abaixo.

```py
class DPCA:

    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit(self, X, Y):

        # Start fitting
        X_pca = self.pca.fit_transform(X)
        svm = LinearSVC().fit(X_pca, Y)

        # Sorting PCA components
        sorted_idx = np.flip(np.argsort(np.abs(svm.coef_)))
        self.pca.components_ = self.pca.components_[sorted_idx.ravel(), :]

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.pca.transform(X)
```

No método `DPCA.fit` foi colocado o código para o algorítmo da DPCA usando uma
SVM linear como classificador. Este método segue o seguinte procedimento:

1. Os dados `X` são usados para computar a matriz de projeção do PCA e
projetados através da mesma (`pca.fit_transfom`); 

2. Uma SVM Linear é treinada nos dados projetados e suas etiquetas (`X_pca`, `Y`);

3. A matrix de projeção da PCA é então modificada para ficar em ordem
decrescente de valor absoluto  das componentes do vetor diretor do hiperplano
separador da SVM.

Assim o método `DPCA.transform` agora irá usar a matriz reordenada para fazer
as projeções, que irá basicamente rodar a projeção do modelo de PCA padrão mas
com a base reordenada. Com isso o DPCA pode ser usado junto com a
funcionalidade de pipeline do sklearn e podemos reutilizar o código usados nos
itens anteriores para fazer a validação cruzada. Podemos então instanciar um modelo 
usando o código:

```py
make_pipeline(
        DPCA(150),
        SVC(kernel="sigmoid")
    )
```

Foram usadas novamente 150 componentes principais e uma SVM com kernel
sigmoidal como na questão 1 pelas mesmas razões. Com isso podemos iniciar a
validação cruzada com k-fold, usando K=4 como solicitado. Fazendo isso foram
obtidas as matrizes de confusão apresentadas na @fig-conf-3b.

#figure(
  image("figures/q3_b_cross_val.png"),
  caption: "Matrizes de confusão da validação cruzada"
) <fig-conf-3b>

O mesmo apresentou um desempenho similar ao do item a, tendo uma alta taxa de
acerto e não foi observado viés nas classificações erradas. No desempenho global 
foi obtido para a acurácia uma média de 96.50% e para o desvio padrão 1.66%

=== (c) Comparando as configurações

As estatísticas obtidas para o desempenho global foram para a configuração (a)
$mu_b = 96.75%$ e $sigma_a = 1.48%$ e  para a configuração (b) foi obtido $mu_b
= 96.50%$ e $sigma_b = 1.66%$. Novamente os pontos apresentam um alto grau de
sobreposição, como pode ser visto na @fig-kde-3c.

#figure(
  image("figures/q3c_kde.png", width: 50%),
  caption: "KDE plot das medidas de acurácia em ambas as configurações"
) <fig-kde-3c>

Dada uma diferença tão pequena entre as distribuições, e considerando que é
ainda menor que o observado na questão 1 como foi argumentado anteriormente,
não é possível afirmar a existência de uma diferença significativa no
desempenho global dos métodos. 

Seria esperado uma melhoria de classificação usando o DPCA, no entanto,
possivelmente por conta da decisão de usar 150 componentes principais para
manter a uniformidade com os testes da questão 1, os benefícios da DPCA podem
ter sido perdidos, pois a técnica minimiza a quantidade de PC's necessárias
para classificação colocando-as no início da matriz, portanto usando muitas
PC's não deve ser observada muita diferença se a as mais discriminates são as
10 primeiras por exemplo.
