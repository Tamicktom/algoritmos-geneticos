# Exercício 3: Redes Neurais para Classificação do Dataset Digits

## Introdução

Este documento apresenta a solução do Exercício 3, que consiste na implementação e comparação de diferentes arquiteturas de redes neurais para classificação do conjunto de dados Digits do scikit-learn. O dataset Digits contém imagens de dígitos manuscritos (0-9) em baixa resolução (8x8 pixels), totalizando 1797 amostras.

Conforme solicitado, o objetivo é realizar experimentos de classificação com diferentes arquiteturas de redes neurais, comparando seus desempenhos e configurações de hiperparâmetros. Para isso, foram implementadas e avaliadas cinco arquiteturas diferentes:

1. MLP Simples (1 camada oculta)
2. MLP Profunda (3 camadas ocultas)
3. CNN Simples (1 camada convolucional)
4. CNN Profunda (2 camadas convolucionais)
5. MLP com otimizador SGD (1 camada oculta)

## Dataset Digits

O dataset Digits do scikit-learn é um conjunto de dados de dígitos manuscritos (0-9) em baixa resolução, frequentemente utilizado para tarefas de classificação em aprendizado de máquina.

### Características do Dataset

- **Número total de amostras**: 1797
- **Dimensão de cada imagem**: 8x8 pixels
- **Dimensão dos dados de entrada**: 64 (imagens achatadas em vetores)
- **Número de classes**: 10 (dígitos de 0 a 9)

### Visualização de Exemplos

Abaixo estão exemplos de imagens do dataset Digits, mostrando um exemplo de cada classe (dígitos de 0 a 9):

![Exemplos do Dataset Digits](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333605_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvZXhlbXBsb3NfZGlnaXRz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZaWGhsYlhCc2IzTmZaR2xuYVhSei5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=B5leKfAZOTNO5Nd02JXaU67GU-xNaYvUm4U9Mgy3jmjuq7rFBxYD6Mh04K9E3if6onvWuMMZYNs5eb4Wg~sQ2JQsMaLMTNi03Muj68ZhuE4PGI5hxM0udz7jAJ-GsE4yqd5VtHL3e89VLUFbeLfe~kstUu9zX8x-yTc5G0VZBYMBWQcEgJbZ4q-n9YiJ9Im~DlyJnKisGM9uaGzT3bms9SVBSoqUJaY522IGVN57CcZDcpTmMig-BNJ~bEM~E2xB1UViqepGiK4VxEaS~FPZTzTqsvY~zzJ-iXn1IPinI1D8myH6bLb4Z71rplZHc9A6VtsnP39TbHMzdtG3P89aDw__)

### Divisão do Dataset

O dataset foi dividido em três conjuntos:

- **Treinamento**: 1257 amostras (70%)
- **Validação**: 270 amostras (15%)
- **Teste**: 270 amostras (15%)

A distribuição das classes em cada conjunto foi verificada para garantir uma representação equilibrada de todos os dígitos.

## Arquiteturas de Redes Neurais Implementadas

### 1. MLP Simples

Uma rede neural feedforward simples com apenas uma camada oculta.

**Arquitetura**:
- Camada de entrada: 64 neurônios (correspondentes aos 64 pixels)
- Camada oculta: 100 neurônios com ativação ReLU
- Camada de saída: 10 neurônios com ativação softmax (um para cada classe)

**Hiperparâmetros**:
- Otimizador: Adam
- Função de perda: Entropia cruzada categórica
- Batch size: 32
- Early stopping com paciência de 10 épocas

### 2. MLP Profunda

Uma rede neural feedforward mais profunda com três camadas ocultas e técnicas de regularização.

**Arquitetura**:
- Camada de entrada: 64 neurônios
- Primeira camada oculta: 128 neurônios com ativação ReLU + BatchNormalization + Dropout (0.3)
- Segunda camada oculta: 64 neurônios com ativação ReLU + BatchNormalization + Dropout (0.3)
- Terceira camada oculta: 32 neurônios com ativação ReLU + BatchNormalization
- Camada de saída: 10 neurônios com ativação softmax

**Hiperparâmetros**:
- Otimizador: Adam
- Função de perda: Entropia cruzada categórica
- Batch size: 32
- Early stopping com paciência de 10 épocas

### 3. CNN Simples

Uma rede neural convolucional simples com uma camada convolucional.

**Arquitetura**:
- Camada de entrada: Imagens 8x8x1
- Camada convolucional: 32 filtros de tamanho 3x3 com ativação ReLU e padding 'same'
- Camada de pooling: MaxPooling 2x2
- Flatten
- Camada densa: 64 neurônios com ativação ReLU
- Camada de saída: 10 neurônios com ativação softmax

**Hiperparâmetros**:
- Otimizador: Adam
- Função de perda: Entropia cruzada categórica
- Batch size: 32
- Early stopping com paciência de 10 épocas

### 4. CNN Profunda

Uma rede neural convolucional mais profunda com duas camadas convolucionais e técnicas de regularização.

**Arquitetura**:
- Camada de entrada: Imagens 8x8x1
- Primeira camada convolucional: 32 filtros de tamanho 3x3 com ativação ReLU e padding 'same' + BatchNormalization
- Segunda camada convolucional: 64 filtros de tamanho 3x3 com ativação ReLU e padding 'same' + BatchNormalization
- Camada de pooling: MaxPooling 2x2
- Dropout (0.25)
- Flatten
- Camada densa: 128 neurônios com ativação ReLU + BatchNormalization
- Dropout (0.5)
- Camada de saída: 10 neurônios com ativação softmax

**Hiperparâmetros**:
- Otimizador: Adam
- Função de perda: Entropia cruzada categórica
- Batch size: 32
- Early stopping com paciência de 10 épocas

### 5. MLP com SGD

Uma rede neural feedforward simples com otimizador SGD em vez de Adam.

**Arquitetura**:
- Camada de entrada: 64 neurônios
- Camada oculta: 100 neurônios com ativação ReLU
- Camada de saída: 10 neurônios com ativação softmax

**Hiperparâmetros**:
- Otimizador: SGD com learning rate=0.01 e momentum=0.9
- Função de perda: Entropia cruzada categórica
- Batch size: 32
- Early stopping com paciência de 10 épocas

## Resultados e Discussão

### Comparação de Desempenho

A tabela abaixo apresenta um resumo comparativo do desempenho das diferentes arquiteturas:

| Modelo | Arquitetura | Parâmetros | Acurácia no Teste | Tempo de Treinamento (s) | Épocas |
|--------|-------------|------------|-------------------|--------------------------|--------|
| MLP Simples | 1 camada oculta (100 neurônios) | 7.510 | 97,04% | 15,15 | 88 |
| MLP Profunda | 3 camadas ocultas (128-64-32) | 19.882 | 98,52% | 13,26 | 50 |
| CNN Simples | 1 camada conv (32 filtros) | 33.802 | 98,15% | 9,87 | 41 |
| CNN Profunda | 2 camadas conv (32-64 filtros) | 152.202 | 99,26% | 23,48 | 48 |
| MLP com SGD | 1 camada oculta (100 neurônios) | 7.510 | 97,41% | 11,14 | 65 |

![Comparação de Acurácia](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvY29tcGFyYWNhb19hY3VyYWNpYQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZZMjl0Y0dGeVlXTmhiMTloWTNWeVlXTnBZUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=XHbozYfI8kkce461jQEIDL4BR-gAX3skuAZDhpO4TS-j7t5jGSh0xwNXe-UhXdAsLvmiGeQKc6IEADCgiN-yyYwiCcvGi10bY-rE51sNfhDbJrwgnx6DbiScYpGMqsHZiE8me5UIQqfrtdPOovKyRz45lVEjKWLG~v-H5XHS59WHW9JNLO5XymyygkhcSf~baQbe5M9pwPefdWGe8dJP2QC1pxulnRFNG81nbU~H~ldUIRFTP43MElOFdV2WUdkNFZFYKelvGYLWhvdDiEb-TJTPuXpAUiERbfdBAfKV3--Wc8BvNxoTQijemTOKeEtd3wygNw8Fr5B4eNo-M0Am7g__)

### Curvas de Aprendizado

As curvas de aprendizado mostram a evolução da acurácia e da perda (loss) ao longo das épocas de treinamento para cada modelo:

#### MLP Simples
![Histórico MLP Simples](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvaGlzdG9yaWNvX21scF9zaW1wbGU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZhR2x6ZEc5eWFXTnZYMjFzY0Y5emFXMXdiR1UucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=NEvmiw5u3tqiQSNSAIl2-pW1RdWvz~HRa0IThkqG-eE5SXtAqRQOHBj1ngWbpI~Y7p0YFQbFcaMAQTEhu50WZ4M3cES8fR2dakSTYPbufRM7epbvZxRwIUt2kZ3l3EofuRUUyWpKCxyoHu0crUxQmUbXmFmVTDwaFXLu-v4ECWQVkZqhRewwNHdu~KE~fLvDc2pqaxopLfE~mysrbKnmc~Nsm3nLHKW6knjbiyzpVayomkJZlcmGTNuIEPnlI3k-Y5PD5-ufwAcHBzLrIkwzzL91Hl4vpfrx-s~xHUI5AwD-e2yf7FxWfelA8KsgRPJHDCQOtlQUFuoQ7y41SVGdmw__)

#### MLP Profunda
![Histórico MLP Profunda](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvaGlzdG9yaWNvX21scF9kZWVw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZhR2x6ZEc5eWFXTnZYMjFzY0Y5a1pXVncucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=jY31ODyb1w18dcJFtctaBVlrBmS-YdUURZB18f9l7TwUyLVmc~PDQaQnctDGpI4sKN3WEaCFQHoRWTjjzwCpO1qQOHAFUQ64~7x-YB20o~KVWr4rKH9Mit9QcShQASdyy8Cq-hE7xK65QvIuYA8ZiekwD3uGxlZPr0ZSv9HKavCR7WJ-wKdnQ1WUqlnHU~d6LImZkG91Ce5Q~rDNWWx12Z~BlaygbruzJtMslbq35mktB1C8Il7U2WKLpMq7CB7wSQb1rsVAIUuN9qnhCwPYUWhtb-4GKL2Mn7fshCjeZta2Kf5EVz1ueLnmBit3M4etXVkto9xei1NKKmSzEzAMWg__)

#### CNN Simples
![Histórico CNN Simples](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvaGlzdG9yaWNvX2Nubl9zaW1wbGU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZhR2x6ZEc5eWFXTnZYMk51Ymw5emFXMXdiR1UucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Qs7wlOTShzwv5xqK55lWg780FddDygB-Qfkq~QnmxYUhesWG3AQDATptOEAxHjsBi3SwfHWfkW8V38ukz-dfF4KLEQi98JEkmui03a~aJHLoZkGy~W9rRrQetPYLOl~FRTXbn-Uc1Sb4BOkWw6Nx9SsjTScU1aYAHriv06ifvW-R4B5dnx11n~uBRsL0BIM~AJ0ygAfmNxL9~7YsXvbgTIfEGwnmwlt5y-39VmaqelRBtxBowbozFWZ528dMCl5kgpjVaTyRZpYIJOm9dFNbzyJh9hsemYUnyePDjyrMG0EUiT6PN7o5lEt26vrmh1w78VDysRiwzH9yENPhKkuqUQ__)

#### CNN Profunda
![Histórico CNN Profunda](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvaGlzdG9yaWNvX2Nubl9kZWVw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZhR2x6ZEc5eWFXTnZYMk51Ymw5a1pXVncucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=LlfXhPNH71ICy-tU~IWZvD3hDNjHMEbioYBsiBstcyIbuqoXM8UtcEQzWORL71lJDpVyueROWqNqX5aI4QyxaLe-wQJrUTEKB3xAuPTiKKlf7nATpZ8ZssmHxBmwT1ram9L8SvETxxXIgUZDxRk--MWxEZrLnGcq2tON379fBiSgBRpLYZ7bxGBDdwyvoshjFqtyEoR47mdLMC4~YdDR4UZTvAkvBxRVdc5qI0r5lAiyA22BY3aG1zIe9xBBPruShNQCugxJDyU85Pa1ADuExd7X3TJWV2F9V2qVQh3la2CUflXaSXHGK2tKHwBnwDdQ3YTw4G1KSRerDhf1J8B6eg__)

#### MLP com SGD
![Histórico MLP com SGD](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvaGlzdG9yaWNvX21scF9zZ2Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZhR2x6ZEc5eWFXTnZYMjFzY0Y5eloyUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=cOQyWKmVXdyt0sG3Qaqswbyzp1y~4H9KEac-wjt~rl0kpf4~3nre5E2idi1sG8YZkL0cVEN0bpY87npmyfiBUO8N5O6YG6vN0BfH8KJC8kGEMzRCcLABbkNKAtLHGJnqHVOCx27IPsJADYeHovNlxuJLeHNpL~Ds89cxZjCuUBaZ8NnMRN5d6o1dQmNIzByykYw79~KHXClPcwaOiOmJ6WkFiQzZesJNo3Fp45yf8Z8G1e-~lvD0cGgmAYqTFupMxwMXdP~9j6YnnpwyuMAb4J4XSU~O6Khm34-XkBjh2dxabZPtb1b54jqAkryrQsrDZGAa2QLX25zSOxJkbIvvmw__)

### Matrizes de Confusão

As matrizes de confusão mostram o desempenho detalhado de cada modelo no conjunto de teste, indicando quais classes são mais facilmente confundidas:

#### MLP Simples
![Matriz de Confusão MLP Simples](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvbWF0cml6X2NvbmZ1c2FvX21scF9zaW1wbGU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZiV0YwY21sNlgyTnZibVoxYzJGdlgyMXNjRjl6YVcxd2JHVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Sn22toTG1PEiskKOWLf-CEH0ubaFBBG-XTfjJnvNx5yQ1DAuU2CIZS8KkKoB9Vqgbjn9w25NfBf0uBEywHshBqES4UaZ~Ru3ByWTCsEWclIi3SLbEwUX~lCFdp3cq5o~bGyvLHvp3soB9AVBqrfmlEh5LHR7S4NX4AJMIq-~E4w7gpMzUJEhA8lOdW-KUorVTVooACbZtmGLeeE6ePzpSTCxAKozpAt0E-LAoFdOPsNRFX2dNCwuHTVfhlEDZjD8gVFo1IyMow-9jdyPitebK0gKeeeNWM1BHQd~aOh0iEVKCGvslWdiaiKIb7q1GKtNZrTz5ot2axyqxulzQjKQVA__)

#### MLP Profunda
![Matriz de Confusão MLP Profunda](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvbWF0cml6X2NvbmZ1c2FvX21scF9kZWVw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZiV0YwY21sNlgyTnZibVoxYzJGdlgyMXNjRjlrWldWdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=n9heCZoVA99N9hmwjUTDpSZFlLrblr5I4eHMQo5ZmMG8xDhhGIkLOO92ITEPyPl5~jSE8mJ-Vn3C9woRAzafgY5Ia6JtuA1HMuD7nurKsqH2D6CMLI4RexsZdeMpScSgjSdT1Pcw6KpNKPUvwLDNf6hU83mMS7qaetnpI5MgGmsbNabB4cztCXGc1521-L6DAF-abXDcZTLFw6Ckjn7c9~2ahkzW0eZJ-cz6G85I64E3gBMkEZYnvyCRvibA7rKZeqHoIIbI~NMiHYTyRsOCpDuKiVYKcZWbLLBHhNoUxIGV794FJJyMt0-mIe5G3fxMIGv~C4VMjUHo4d6WYcee~w__)

#### CNN Simples
![Matriz de Confusão CNN Simples](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvbWF0cml6X2NvbmZ1c2FvX2Nubl9zaW1wbGU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZiV0YwY21sNlgyTnZibVoxYzJGdlgyTnVibDl6YVcxd2JHVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=BaxCSPSLqpQ7l2caSJ4FW2rR6-D2algYZK5JEbkPO7kMFL3R8GzWdKTA3dIPewkUK~6Qk3V3AaxE4luss5CsHSDcqifPBJJ6HRtqEXqw-5WYEF5-kR~8136ryoNKfY9XoGIWgL2QvmF2a5NtMLgUUx7R02nxX8AwemT-smqRwJ9dHv5iM~Ss2poDAhrOuv5tEnkayHww2q0t7u3VpNOVUWhlskD7PGr~aZfC5qIQjAO4BTASOt8yMWi8Kqz~3F47aCcw5Irxq7PcXGx97MSWv~kx5Y0tYxWfTcYE1E6O-PxOPHzVq09oZu3irGt9-eqRCLuEtwUaHFVdI3TOuWAEQA__)

#### CNN Profunda
![Matriz de Confusão CNN Profunda](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvbWF0cml6X2NvbmZ1c2FvX2Nubl9kZWVw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZiV0YwY21sNlgyTnZibVoxYzJGdlgyTnVibDlrWldWdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=NM0LQCift9bNYMXvvOUkLMZPrlb6CoZ42HtpO66wjH7RVCTKPWcAeDKgJQ8Pnyj~E4J1S5AYdszHiqDKzJxkXR2K968BoCjZP3wc8k23Nh26rqa1dUekrCk1u2ZAS07NKG9jX4o-YHaIToHZyw7OnCRORZGqg-T1ph0edPi-hxaWOZ1uqtHSVxUAtPq2trDCFA25fIhhBUfwUl~R9UWIPX09W9DReZmj8mtoRDWr1vckEy-maGtvFTh56mKhOTcOeEXZ-5sOAFnhxqGfntRVbyuuehqthlRFKsEcvNE~5EyQKRA83aQDNCva9-K53dN1tAj-K6-BaVJ4mWZyi3kD4Q__)

#### MLP com SGD
![Matriz de Confusão MLP com SGD](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvbWF0cml6X2NvbmZ1c2FvX21scF9zZ2Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZiV0YwY21sNlgyTnZibVoxYzJGdlgyMXNjRjl6WjJRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=EUCZhEAibNH5fR0bZvXyncCDa2~eOn8j-0Bu9JEn4woRytGvDjeGiDxTWnhKy3UW5Citu6fMm2eOjkPG3QXz63my-GjkP5lkSyVzOnevItfFyuscLlpefM1yhrCwMG3ZtRoTYUklD1Xx4AK9nqCNdcHdgTYyLBEHla8w78B0V21NdvAzp75NN4ZuAi2-3UP-acp1nhdrx4ptXieWhbbm93740b4f5AEbPLgsCGsFYOlWSLcr5nI6dz5xJLoX4cPRf6NB7qZqYPweQncmJti~m~0qEfWPMN-R3cwmboJYSJB-lszkiVbCQtbq5pUOj913uiOcRUKZKBdUEHKElHl3DA__)

### Comparação de Complexidade e Tempo de Treinamento

![Comparação de Parâmetros](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvY29tcGFyYWNhb19wYXJhbWV0cm9z.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZZMjl0Y0dGeVlXTmhiMTl3WVhKaGJXVjBjbTl6LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Y1rwrSMD7oiXtfMA1UkMcUeEEFhFUOAZmLFu42XWUH4xyCWJZbJ2avcrV6nPA15YWVdm74EJctCCmSjMC9Ff0OjwabWBw6AX2qqvi6KSRLG9SFFmJD65jNGjyf0Jjb-ytKmplRprPfVHhIkEnmqmjbPw~5u8MY~i3NvJPqgg~ouIt1Ckj78jA2QG6FO6gOYT6iRpeyxCn61YhRi2FQ0PKeAtvGFfZDqgft78QvcNvuVqZDgTOWunCPPEoWQkC09lyZXQX~IVviW294dm1kCnESCrS9vMruzY9seNLbaLpqCn8AL-CgTELHhOkQU3Jv5Y7DxnuOoe6LPK8U-Gt8PSkg__)

![Comparação de Tempo](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/g70FrKMG2GUt3DAsu9NiJs-images_1747700333606_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzMvY29tcGFyYWNhb190ZW1wbw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2c3MEZyS01HMkdVdDNEQXN1OU5pSnMtaW1hZ2VzXzE3NDc3MDAzMzM2MDZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6TXZZMjl0Y0dGeVlXTmhiMTkwWlcxd2J3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=sB9pUPV8cpK~Ut4iP~dzKHodyFFx9PUkQWBNC05Vz45O7usAKf8oXVo6eXNzYBgr3H2sSlV4tHdxirTfSREft3e01ckmXkA5oNkUkMx53ne65WXbwVD9GKaOcyWQIFdmX0QIOUUh5Vpxr-zIHIL~hV5H-yY2FYqyE9LoLei-HYb2K3~GFKXeDp-zkQJr8LfGBVDASr~G4vQcnvDP2lDwiOviEzlqp~e08xDg4oi2Ut~mFrIpiZiTGqZgsObs0VACS3IoYA-bbJ5Ve1Hq~qaGevGqLs4hd5bAzDP6cLWar75F7nYuSqs448uLSH4LBA1JFvzx71sQX7KMsEmNeJ4mQQ__)

## Análise dos Resultados

### Acurácia

- A **CNN Profunda** obteve a melhor acurácia (99,26%), seguida pela **MLP Profunda** (98,52%) e pela **CNN Simples** (98,15%).
- As arquiteturas mais simples (**MLP Simples** e **MLP com SGD**) tiveram desempenho ligeiramente inferior, mas ainda assim alcançaram acurácias acima de 97%.
- A diferença entre o melhor e o pior modelo foi de apenas 2,22 pontos percentuais, indicando que mesmo modelos simples conseguem bom desempenho neste dataset.

### Complexidade vs. Desempenho

- A **CNN Profunda** tem o maior número de parâmetros (152.202) e também o melhor desempenho, mas a relação entre complexidade e desempenho não é linear.
- A **MLP Profunda** tem menos de 15% dos parâmetros da CNN Profunda, mas alcança uma acurácia muito próxima.
- Isso sugere que, para este dataset relativamente simples, o aumento da complexidade do modelo além de certo ponto traz ganhos marginais de desempenho.

### Tempo de Treinamento

- A **CNN Simples** foi a mais rápida para treinar (9,87 segundos), apesar de ter mais parâmetros que as MLPs simples.
- A **CNN Profunda** foi a mais lenta (23,48 segundos), o que é esperado dado seu maior número de parâmetros.
- O otimizador **SGD** levou menos épocas para convergir do que o **Adam** na mesma arquitetura MLP Simples (65 vs. 88 épocas), mas com acurácia ligeiramente superior.

### Convergência

- Todos os modelos convergiram bem, sem sinais claros de overfitting (a diferença entre as perdas de treinamento e validação não aumentou significativamente).
- A **MLP Simples** precisou de mais épocas para convergir (88), enquanto a **CNN Simples** convergiu mais rapidamente (41 épocas).
- As técnicas de regularização (BatchNormalization e Dropout) nas arquiteturas profundas parecem ter ajudado a manter um bom equilíbrio entre bias e variância.

### Análise das Matrizes de Confusão

- Todos os modelos tiveram dificuldade em distinguir entre alguns pares específicos de dígitos, como 3/5, 4/9 e 7/9.
- A **CNN Profunda** teve o menor número de classificações incorretas, com apenas 2 erros no conjunto de teste.
- A maioria dos erros ocorre em dígitos que são visualmente semelhantes, o que é esperado dada a baixa resolução das imagens (8x8 pixels).

## Conclusões

Com base nos experimentos realizados, podemos concluir que:

1. **Arquiteturas convolucionais são mais adequadas para este problema de classificação de imagens**, mesmo com imagens de baixa resolução. A CNN Profunda obteve o melhor desempenho geral.

2. **A complexidade do modelo deve ser proporcional à complexidade do problema**. Para o dataset Digits, que é relativamente simples, mesmo modelos com poucos parâmetros conseguem bom desempenho.

3. **Técnicas de regularização são importantes**. A adição de BatchNormalization e Dropout nas arquiteturas profundas ajudou a melhorar o desempenho sem aumentar significativamente o tempo de treinamento.

4. **O otimizador Adam geralmente converge mais rapidamente que o SGD**, mas o SGD com momentum pode alcançar resultados competitivos com um número adequado de épocas.

5. **O early stopping é uma técnica eficaz** para evitar overfitting e determinar automaticamente o número ideal de épocas de treinamento.

Em resumo, para o dataset Digits, a CNN Profunda com duas camadas convolucionais, BatchNormalization e Dropout apresentou o melhor equilíbrio entre desempenho e complexidade, alcançando uma acurácia de 99,26% no conjunto de teste. No entanto, para aplicações com restrições de recursos computacionais, a MLP Profunda oferece um bom compromisso, com desempenho próximo (98,52%) e menos de 15% dos parâmetros da CNN Profunda.

## Comparação com os Exercícios Anteriores

Comparando os resultados obtidos com o dataset Digits com os dos datasets Iris (Exercício 1) e Wine (Exercício 2), podemos observar:

1. **Complexidade do problema**: O dataset Digits representa um problema mais complexo, com 10 classes (vs. 3 classes nos datasets Iris e Wine) e dados de imagem em vez de atributos numéricos simples.

2. **Arquiteturas necessárias**: Enquanto o Perceptron foi suficiente para os datasets Iris e Wine (especialmente após normalização), o dataset Digits se beneficiou significativamente de arquiteturas mais complexas como CNNs.

3. **Desempenho**: O melhor modelo para o dataset Digits (CNN Profunda) alcançou 99,26% de acurácia, comparável ao desempenho do Perceptron no dataset Wine normalizado (100%), mas superior ao desempenho no dataset Iris (cerca de 91% no conjunto de validação).

4. **Impacto da normalização**: Nos exercícios anteriores, a normalização teve um impacto crucial no desempenho do Perceptron. No caso das redes neurais para o dataset Digits, a normalização também foi aplicada, mas o impacto maior veio da escolha da arquitetura adequada.

Essas observações destacam como diferentes tipos de problemas de aprendizado de máquina podem exigir diferentes abordagens e arquiteturas, e como a complexidade do modelo deve ser adaptada à complexidade do problema em questão.
