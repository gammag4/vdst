# View-Depth Synthesis Transformer: Um modelo baseado em Transformer para RGB-D Novel View Synthesis

| [English](README.md) | Português |

Essa é a implementação do modelo VDST, junto com o código para treiná-lo com os datasets usados originalmente.

Este é um modelo de RGB-D Novel View Synthesis, onde dadas um conjunto de visões de uma cena 3D com respectivos mapas de distâncias e propriedades/poses de câmera destas,
o modelo busca gerar uma nova visão com respectivo mapa de distância na cena, dadas as propriedades e pose da visão que se deseja gerar.

Esse modelo foi inspirado primariamente pela arquitetura e filosofia de [LVSM](https://haian-jin.github.io/projects/LVSM/),
onde ele aplica a mesma ideia central para a tarefa de Novel View Synthesis RGB-D.

### Exemplo de cena renderizada

Cena renderizada:

<https://github.com/user-attachments/assets/35a33c5e-fd6a-473b-b55a-ea6f4a667a93>

Imagens-fonte:

![Imagens-fonte](final_results/render/sources.png)

### Arquitetura

<img src="Architecture.png" width="800px" alt="Arquitetura">

#### Processo

Entradas:
- Imagens $I_k \in [0, 1]^{3 \times W \times H}$;
- Mapas de profundidade $D_k \in \mathbb{R}_+^{W \times H}$;
- Máscaras de profundidade $M_k \in \{ 0, 1 \}^{W \times H}$;
- Matrizes de intrínsecos $K_k \in \mathbb{R}^{3 \times 3}$;
- Matrizes de extrínsecos $E_k \in \mathbb{R}^{4 \times 4}$;

Saídas :
- Imagem-alvo $I_{\mathrm{target}} \in [0, 1]^{3 \times W \times H}$;
- Mapa-de-profundidade-alvo $D_{\mathrm{target}} \in \mathbb{R}_+^{W \times H}$;

As imagens-alvo são dadas por:

$$
\begin{align}
[I_k, D_k] &= \left[\frac{I_{k, \mathrm{raw}} - i_{\mu}}{i_{\sigma} + \varepsilon}, \frac{\log(D_{k, \mathrm{raw}} + \varepsilon) - d_{\mu}}{d_{\sigma} + \varepsilon} \right] & \text{ (normalized sources)} \\
\begin{bmatrix} R_k & \mathbf{t}_k \\ 0 & 1 \end{bmatrix} &= E_k & \text{ (rotation/translation)} \\
\mathbf{v}_{ij} &= [i + 0.5, j + 0.5, 1] & \text{ (raw pixel center vectors)} \\
[\mathbf{o}_{k, ij}, \mathbf{d}_{k, ij}] &= [\mathbf{t}_k, R_k K_k^{-1} \mathbf{v}_{ij}] & \text{ (inverse projection (c2w))} \\
O_{k, ij} &= [\mathbf{d}_{k, ij}, \mathbf{o}_{k, ij} \times \mathbf{d}_{k, ij}] & \text{ (Plücker embedding)} \\
\mathrm{Tokenize}(P) &= P W_{P} + b_P & \text{ (linearly maps to tokens)} \\
X &= \mathrm{Tokenize}(\mathrm{Patchify}([[O_1, I_1, D_1], \dots, [O_N, I_N, D_N]])) & \text{ (concat, patchify and tokenize sources)} \\
Q &= \mathrm{Tokenize}(\mathrm{Patchify}([O_1, \dots, O_N])) & \text{ (concat, patchify and tokenize queries)} \\
[X_{discard}, Y] &= \mathrm{Transformer}([X, Q]) & \text{ (output tokens)} \\
[I_{\mathrm{target}, \mathrm{raw}}, D_{\mathrm{target}, \mathrm{raw}}] &= \mathrm{Depatchify}(\mathrm{Detokenize}(Y)) & \text{ (normalized targets)} \\
[I_{\mathrm{target}}, D_{\mathrm{target}}] &= [i_{\sigma} I_{\mathrm{target}, \mathrm{raw}} + i_{\mu}, e^{d_{\sigma} D_{\mathrm{target}, \mathrm{raw}} + d_{\mu}}] & \text{ (denormalized targets)}, \\
\end{align}
$$

onde $\mathrm{Transformer}(\cdot)$ é o transformer, implementado com pre-layer normalization nos blocos residuais e query-key normalization na atenção.

A perda é dada por:

$$
\begin{align}
\mathcal{L}([I_{\mathrm{target}}, D_{\mathrm{target}}], [\hat{I}_{\mathrm{target}}, \hat{D}_{\mathrm{target}}]) &= \mathcal{L}_I(I_{\mathrm{target}}, \hat{I}_{\mathrm{target}}) + \mathcal{L}_D(D_{\mathrm{target}}, \hat{D}_{\mathrm{target}}) \\
\mathcal{L}_{I}(I, \hat{I}) &= \lambda_{I, \mathrm{MSE}} \mathcal{L}_{I, \mathrm{MSE}}(I, \hat{I}) + \lambda_{I, \mathrm{Percep}} \mathcal{L}_{I, \mathrm{Percep}}(I, \hat{I}) \\
\mathcal{L}_{D}(D, \hat{D}) &= \lambda_{D, \mathrm{SILog}} \mathcal{L}_{D, \mathrm{SILog}}(D, \hat{D}) + \lambda_{D, \mathrm{grad}} \mathcal{L}_{D, \mathrm{grad}}(D, \hat{D}) + \lambda_{D, \mathrm{Percep}} \mathcal{L}_{D, \mathrm{Percep}}(D, \hat{D}) \\
\mathcal{L}_{I, \mathrm{MSE}}(I, \hat{I}) &= \mathrm{MSE}(I, \hat{I}) \\
\mathcal{L}_{I, \mathrm{Percep}}(I, \hat{I}) &= \frac{1}{N_{\mathrm{layers}} C_l H_l W_l} \sum_{l = 1}^{N_{\mathrm{layers}}} \lambda_l \left\| \phi_l\left( \frac{I - \mu_I}{\sigma_I} \right) - \phi_l\left( \frac{\hat{I} - \mu_I}{\sigma_I} \right) \right\|_1 \\
\mathcal{L}_{D, \mathrm{SILog}}(D, \hat{D}) &= \mathrm{MSE}(\log(D), \log(\hat{D})) - \lambda \mathrm{ME}(\log(D), \log(\hat{D}))^2 \\
\mathcal{L}_{D, \mathrm{Grad}}(D, \hat{D}) &= \frac{1}{N_{\mathrm{scale}} C_l H_l W_l} \sum_{l = 1}^{N_{\mathrm{scale}}} |\nabla_X (D_l - \hat{D}_l)| + |\nabla_Y (D_l - \hat{D}_l)| \\
\mathcal{L}_{D, \mathrm{Percep}}(D, \hat{D}) &= \mathcal{L}_{I, \mathrm{Percep}} \left( \frac{\log(D) - \log(\hat{D})}{\sqrt{2} \sigma_{\log, D}}, \mathbf{0} \right), \\
\end{align}
$$

onde cada $\phi_l(\cdot)$ nas perdas perceptuais é o mapeamento de cada camada $l$ das camadas escolhidas da ConvNeXt,
$\mu_{D, Percep} = \mu_{\log, D} - \mu_{\log, D} = 0$, $\sigma_{D, Percep} = \sqrt{\sigma^2_{\log, D} + \sigma^2_{\log, D}} = \sqrt{2} \sigma_{\log, D}$,
e $\mu_I, \sigma_I, \mu_{\log, D}, \sigma_{\log, D}$ são as médias e desvios padrões das imagens e profundidades-log calculadas para todo o dataset.

### Resultados

Aqui estão alguns dos resultados do modelo depois de treinado com batch size 4 por 350000 iterações usando uma RTX 4060 Ti com 8GB de VRAM, durando em torno de 153 horas.

Número de parâmetros: 43.46M.

Métricas:

- Imagem:
  - PSNR: 19.72
  - SSIM: 0.588
  - LPIPS: 0.424
- Profundidade:
  - AbsRel: 0.0401
  - RMSE: 0.168
  - $\delta_{1}$: 0.971

Conjunto de validação:

![Resultados no conjunto de validação](final_results/val/targets_0_val.png)

Conjunto de teste:

![Resultados no conjunto de teste](final_results/test/targets_0_val.png)

Conjunto de teste nova categoria (categoria `truck`):

![Resultados no conjunto de teste nova categoria](final_results/test_new_category/targets_0_val.png)

### Contribuições e limitações

Principais contribuições:

- A arquitetura (VDST);
- Uma variação da perda perceptual usando ConvNeXt no lugar de VGG-19;
- Uma perda perceptual para profundidade baseada em ConvNeXt com uma segunda CNN específica para estimar o campo receptivo e permitir usar máscaras de valores inválidos nos latentes da perda;
- Um pipeline de processamento de datasets de cenas RGB-D que permite isolar a avaliação de geração de conteúdo e interpolação de visões nas novas visões geradas.

Vantagens principais que esse modelo tem em comparação com outros métodos de NVS RGB-D:

- Devido à capacidade de generalização entre domínios de modelos baseados em Transformer, ele é capaz de:
  - Generalizar para cenas novas que seguem uma distribuição similar à dos dados originais de treino;
  - Realizar few-shot NVS, precisando de apenas duas imagens-fonte ou às vezes apenas uma na maioria dos casos para gerar novas visões;
- Seguindo a mesma abordagem de LVSM, nossa arquitetura também tenta minimizar o viés indutivo do modelo,
  e hipotetizamos que isso permite alcançar melhor resultados do que outros métodos quando treinado por períodos mais longos com quantidades suficientemente grandes de dados,
  apesar de não termos recursos computacionais suficientes para verificar isso, deixando tal investigação para trabalhos futuros;
- Ele pode ser treinado com recursos limitados sem divergir (o autor usou uma única RTX 4060 Ti com 8 GB de VRAM).

Principal limitação do modelo:
Esse modelo foi treinado sob restrições pesadas de recursos, onde, como afirmado anteriormente, tivemos apenas uma NVIDIA RTX 4060 Ti 8GB VRAM disponível para treinar o modelo, um ambiente mais restrito até que o de outros modelos de síntese conjunta de visão e profundidade generalizável treinados sob restrições de recursos, que comumente usam uma NVIDIA A100 80GB VRAM ou similar.
Isso nos forçou a usar um modelo bem menor em comparação com outros (nosso modelo tem 43.5M parâmetros, enquanto outros modelos como [LVSM](https://haian-jin.github.io/projects/LVSM/) ou [MVGD](https://mvgd.github.io/) têm em torno de 400-600M parâmetros), e nos impediu de treinar o modelo até convergência, o que causou ele a ainda ter alguns artefatos visuais.

## Inferência

Os pesos do modelo treinado podem ser encontrados [aqui](https://huggingface.co/gammag7/vdst).

Nós também fizemos um renderizador que você pode usar para navegar nas cenas usando esse modelo, você pode encontrá-lo [aqui](https://github.com/gammag4/nvs_renderer).

## Dataset

Um script para baixar e processar o WildRGB-D dataset pode ser encontrado [aqui](https://github.com/gammag4/nvs_datasets).

O dataset foi dividido em quatro conjuntos:

- `train`: Usado para treinar o modelo;
- `val`: Usado para avaliar as métricas durante e no final do treino dos experimentos e escolher o melhor experimento;
- `test`: Usado para avaliar as métricas do modelo treinado final em cenas novas que não estavam nos conjuntos de treino ou validação;
- `test_new_category`: Usado para avaliar as métricas do modelo treinado final em cenas de uma categoria não usada durante o treino.

Os resultados dos três conjuntos de validação/teste escolhidos estão na pasta `final_results`, cujos resultados estão separados nos seguintes arquivos:

- `scenes.yaml`: Cenas usadas na avaliação quantitativa daquele conjunto;
- `scenes_rendered.yaml`: Cenas escolhidas daquele conjunto para serem renderizadas e usadas na avaliação qualitativa;
- `sources_n_val.png`: Imagens-fonte do batch n usado para avaliação qualitativa, cujas imagens são especificadas por `scenes_rendered.yaml`;
- `targets_0_val.png`: Renderizações das imagens-alvo do batch n usado para avaliação qualitativa, cujas imagens são especificadas por `scenes_rendered.yaml`;
- `eval_metrics.yaml`: Métricas finais da avaliação quantitativa (`snr_log` e `sqrt_snr_log` são apenas para debug, não são métricas reais).

## Treinamento

### Requisitos

Você precisa de:

- Alguma distribuição conda (recomendamos usar [Miniforge](https://conda-forge.org/download/))
- NVIDIA drivers com suporte para CUDA >= 13.0

### Baixando e processando datasets

Baixe e processe o dataset WildRGB-D usando o script disponibilizado para a pasta `datasets/wildrgbd`.

### Criando ambiente

Crie o ambiente Python e instale as dependências:

```sh
conda create -n vdst python=3.13
conda activate vdst
pip install -r requirements.txt
```

### Treinando o modelo

Rode o script de treino:

```bash
torchrun --standalone --nproc-per-node=gpu train.py --config config.yaml
```
