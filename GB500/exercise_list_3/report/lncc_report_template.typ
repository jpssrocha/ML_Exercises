// Typst template for making simple reports for the disciplines in the LNCC
// masters/PhD program.


#let template(
  title: "Titulo aqui",
  author: "João Pedro dos S. Rocha",
  city-state: "Petrópolis-RJ",
  discipline: "Disciplina",
  professor: "Professor",
  bibliography_file: "references.bib",
  doc
  )  = {


  set document(
    title: title,
    author: author
  )

  // IMPORTS //

  import "@preview/subpar:0.1.1"
  import "@preview/codly:1.0.0": *


  // IMPORTS CONFIGS //

  // codly --> Nicer code blocks
  show: codly-init.with()

  codly(
    number-format: none,
    zebra-fill: none,
    radius: 10pt,
    stroke: 2pt + luma(200)
  )

  // SETTINGS //


  set page(
    paper: "a4",
  )


  // Give latex feel
  set page(paper: "a4")
  set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)
  set text(font: "New Computer Modern", lang: "pt", size: 12pt)


  set heading(
    numbering: "1. ",
  )

  set math.equation(
    numbering: "(1)",
  )


  // SHOW RULES //

  show link: underline

  show raw: set text(font: "New Computer Modern Mono")
  show par: set block(spacing: 1.25em)
  show heading: set block(above: 1.4em, below: 1em)

  //  -------- TITLE PAGES  ------------------------- //

  align(center, text(12pt)[
    #image("figures/lncc.png")
    *Laboratório Nacional de Computação Científica \
    Programa de Pós-Graduação em Modelagem Computacional \
    #discipline
    *
  ])


  v(7cm)


  align(center, text(14pt)[
    #title
  ])

  v(4cm)

  align(right, text(14pt)[
    #author
  ])



  align(bottom + center)[
    #city-state \
    #datetime.today().year()
  ]

  pagebreak(weak: true)

  align(center, text(12pt)[
    #image("figures/lncc.png")
    *Laboratório Nacional de Computação Científica \
    Programa de Pós-Graduação em Modelagem Computacional \
    #discipline
    *
  ])


  v(7cm)


  align(center, text(14pt)[
    #title
  ])

  v(3cm)

  grid(
    columns: (1fr, 2fr),
      align(start)[
      ],
      align(end, text(11pt)[
        Trabalho  apresentado  como  parte  dos  critérios de
        avaliação da disciplina: #discipline \
        Prof: #professor
      ]))

  align(bottom + center)[
    #city-state \
    #datetime.today().year()
  ]

  pagebreak(weak: true)
  //  -------- OUTLINE -------------------------- //

  show outline.entry: entry => {
    entry
    v(0.1cm, weak: true)
  }
  outline(indent: true)
  pagebreak(weak: true)

  set page(
    numbering: "1"
  )

  //  -------- CONTENT START ----------------- //

  [#doc]


  pagebreak(weak: true)
  bibliography(bibliography_file)
}


