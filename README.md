# Iaface — com qual ator você se parece?

App que roda no seu computador: abre a câmera no navegador, tira uma foto e
mostra o **top 3 de atores** mais parecidos com você — junto com a **cartela de
cores** que combina com o seu tom de pele, cabelo e olhos.

Nenhuma foto sai da sua máquina: a câmera, o servidor e o modelo rodam todos em
`localhost`.

## O que o app responde

**1. Com quem você se parece.** Três atores, com a porcentagem, uma etiqueta em
português ("parecido", "pouca semelhança"…) e o valor bruto do cosseno para
quem quiser conferir o número real.

**2. O que a comparação significa.** O app mostra o recorte exato que foi
analisado, um veredito que leva em conta a distância entre o 1º e o 2º lugar
("empate técnico entre X e Y") e avisos sobre a qualidade da foto — desfoque,
pouca luz, rosto pequeno demais.

**3. O que combina com você.** As cores medidas na sua foto, a cartela sazonal
correspondente, peças concretas (cachecol ferrugem, camisa azul-serenity…) e as
cores que é melhor evitar perto do rosto.

## Como a comparação é feita

1. **MTCNN** encontra o rosto e o alinha pelos olhos, nariz e boca. Se não
   achar nada, o app tenta de novo com contraste corrigido e com a imagem
   girada — foto de celular sem EXIF é o motivo mais comum de falha.
2. **InceptionResnetV1** (treinada no VGGFace2) transforma o recorte alinhado em
   um vetor de 512 números. O rosto é medido duas vezes, normal e espelhado, e a
   média das duas leituras é o que vai para a comparação.
3. **Similaridade de cosseno** entre o seu vetor e o de cada ator. Quanto mais
   perto de 1, mais parecidos os rostos.

O modelo não foi treinado para dizer "você é o fulano" — ele mede proximidade
entre rostos. É uma brincadeira, não uma identificação.

### Sobre a porcentagem

O cosseno útil fica entre ~0,10 e ~0,75, e o app estica essa faixa para 0–100%
só para o número ficar legível. **Não é probabilidade.** Como referência: duas
fotos da mesma pessoa passam de 80%; "parecido de verdade" costuma ficar entre
40% e 60%.

### Sobre a cartela de cores

Não vem do ator e não é chute do modelo: são medidas de cor tiradas da sua
própria foto e classificadas pela análise sazonal usada em consultoria de
imagem.

| Medida | De onde vem | Para que serve |
| --- | --- | --- |
| **Subtom** (quente/frio/neutro) | ângulo de matiz da pele em CIELAB | escolhe entre cartelas quentes e frias |
| **Profundidade** (clara/média/profunda) | ângulo ITA°, padrão em dermatologia | separa cores claras de cores encorpadas |
| **Contraste** | diferença de luminosidade entre cabelo e pele | decide entre contraste alto e tom sobre tom |

A pele é amostrada nas duas bochechas e na testa, com a faixa central de
luminância — assim sombra de um lado e brilho do outro não distorcem a medida.
Antes disso, um balanço de branco corrige a dominante da luz do ambiente
(lâmpada amarela deixaria todo mundo "quente").

É uma heurística, não um laudo: o app mostra a confiança da leitura e avisa
quando não conseguiu medir o cabelo ou quando o subtom ficou em cima do muro.

## Instalação

Precisa de Python **3.10 ou mais novo** — inclusive 3.13 e 3.14.

```bash
git clone <este-repo> && cd Iaface
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install --no-deps facenet-pytorch==2.6.0
```

São **dois** comandos de instalação, e o segundo tem o `--no-deps` de
propósito: o `facenet-pytorch` declara `torch <2.3.0`, uma versão que só tem
instalador até o Python 3.12. O código dele funciona bem com o PyTorch atual —
a suíte de testes passa igual nos dois —, então instalamos sem as dependências
declaradas e fixamos as versões no `requirements.txt`. Os scripts `run.sh` e
`run.bat` já fazem os dois passos sozinhos.

A primeira execução baixa ~110 MB de pesos do modelo (uma vez só).

## 1) Montar a base de atores

A base **não** vem no repositório: o script baixa as fotos da Wikipédia na hora.

```bash
./.venv/bin/python -m scripts.build_actors
```

Leva alguns minutos (≈180 atores × 4 fotos). Gera `data/actors.npz` e as
miniaturas em `static/actors/`.

Opções úteis:

```bash
--limit 20          # só os 20 primeiros, para testar rápido
--per-actor 6       # mais fotos por ator = comparação mais estável
--photos-dir fotos  # usa fotos suas em vez de baixar da internet
```

No modo `--photos-dir`, organize assim:

```
fotos/
  Tom Hanks/1.jpg
  Tom Hanks/2.jpg
  Fernanda Torres/1.jpg
```

Para trocar quem entra na comparação, edite a lista em `data/actors.py` (use o
título do artigo na Wikipédia em inglês) e rode o build de novo.

## 2) Rodar

No Windows, dê dois cliques em `run.bat` (ou rode `run.bat` no terminal). No
Mac e no Linux:

```bash
./run.sh
```

Os dois fazem a mesma coisa: criam o ambiente, instalam as dependências,
montam a base se ela não existir e sobem o servidor. Na mão, se preferir:

```bash
./.venv/bin/uvicorn app.server:app --port 8000     # Mac e Linux
.venv\Scripts\uvicorn app.server:app --port 8000    # Windows
```

Abra **http://localhost:8000**, clique em *Abrir câmera*, permita o acesso e
tire a foto. Também dá para enviar uma imagem do disco.

> A câmera só funciona em `localhost` ou HTTPS — é regra do navegador. Se você
> abrir pelo IP da máquina (`http://192.168...`), o botão da câmera falha; nesse
> caso use o envio de arquivo ou um túnel HTTPS.

## Testes

```bash
./.venv/bin/pip install pytest httpx
./.venv/bin/python -m pytest
```

A suíte foi verificada nas duas pontas do intervalo suportado: com o PyTorch
2.2 (o que o facenet-pytorch pede) e com o 2.13 + numpy 2 — 45 testes passando
nos dois, com resultados idênticos.

Os testes de `imaging`, `style` e `matching` rodam em milissegundos, sem tocar
no modelo. Os de `test_api.py` carregam a rede de verdade e se pulam sozinhos se
os pesos ainda não estiverem em cache.

## Estrutura

| Arquivo | O que faz |
| --- | --- |
| `app/config.py` | limiares e caminhos, todos em um lugar |
| `app/imaging.py` | conversão de cor (CIELAB), amostragem e medida de foco |
| `app/face.py` | detecção, qualidade da foto e embedding |
| `app/style.py` | análise de subtom, profundidade, contraste e cartela |
| `app/matching.py` | base de atores, ranking e leitura do resultado |
| `app/server.py` | API FastAPI (`/api/match`, `/api/status`) |
| `scripts/build_actors.py` | baixa fotos, calcula embeddings e grava a base |
| `data/actors.py` | lista de atores de referência |
| `static/` | página, estilo e JS da câmera |

## Limitações

- A base só compara com os atores que você montou — quem não está na lista nunca
  vai aparecer.
- Rosto de lado, óculos escuros ou luz ruim derrubam a qualidade do resultado. O
  app avisa quando detecta isso, mas ainda assim responde.
- A cartela de cores depende de enxergar o cabelo: se a foto corta o topo da
  cabeça, o app usa só pele e olhos e marca a confiança como mais baixa.
- Uma foto por ator já funciona, mas 4–6 deixam o resultado bem mais estável.
