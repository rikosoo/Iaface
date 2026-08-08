# Iaface — com qual ator você se parece?

App que roda no seu computador: abre a câmera no navegador, tira uma foto e
mostra o **top 3 de atores** mais parecidos com você — junto com a **cartela de
cores** que combina com o seu tom de pele, cabelo e olhos.

Nenhuma foto sai da sua máquina: a câmera, o servidor e o modelo rodam todos em
`localhost`.

## Privacidade

**A foto é processada em memória e descartada. Nada é gravado em disco.**

Uma foto de rosto analisada para medir semelhança é dado biométrico, que no
GDPR entra no **Artigo 9** — a categoria especial. O app é construído para não
reter nada:

- A imagem chega como **corpo cru da requisição**, não como multipart. Essa é
  uma decisão de projeto, não um detalhe: o parser de formulário do Starlette
  grava uploads acima de 1 MB num arquivo temporário em disco, e uma foto de
  celular passa fácil desse tamanho. Por isso o `python-multipart` nem está
  entre as dependências.
- O vetor facial não é guardado depois da resposta.
- A resposta vai com `Cache-Control: no-store`, então o navegador não guarda o
  recorte do rosto.
- Sem cookies, sem armazenamento local, sem analytics, sem chamada externa.
- O log do servidor registra `POST /api/match 200` — método, caminho e código,
  nunca a imagem.

Isso é verificável: `tests/test_privacy.py` roda uma análise de verdade e
confere que nenhum arquivo novo apareceu no disco e que nenhum arquivo
temporário foi aberto durante o processo, inclusive com uma foto acima de 1 MB.

A página `/privacidade` explica tudo isso para quem for usar o app, e o aviso
aparece na própria tela de captura.

> Se você hospedar isto na internet em vez de rodar em `localhost`, **você
> passa a ser o controlador dos dados**: precisa de base legal para tratar dado
> do Art. 9 (na prática, consentimento explícito), aviso de privacidade
> próprio, registro de tratamento e provavelmente uma DPIA. O código não gravar
> nada ajuda, mas não substitui nenhuma dessas obrigações.

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

Você **não** precisa baixar fotos nem colar links: o script resolve tudo
sozinho a partir dos nomes. No fim ele imprime um relatório dizendo quantos
atores entraram e quais ficaram de fora.

Opções úteis:

```bash
--limit 20          # só os 20 primeiros, para testar rápido
--per-actor 6       # mais fotos por ator = comparação mais estável
--merge             # soma à base existente em vez de refazer do zero
--photos-dir fotos  # usa fotos suas em vez de baixar da internet
```

### Ritmo e a política de robô da Wikimedia

A Wikimedia corta acesso automatizado que não siga as regras dela, respondendo
`HTTP 429` — inclusive em volume baixo. O script já se comporta: manda um
User-Agent com contato, pede **miniatura** em vez do arquivo original (a
Wikimedia pede isso explicitamente), respeita um intervalo mínimo entre
requisições e obedece ao cabeçalho `Retry-After` quando o servidor manda
esperar.

Se ainda assim aparecer 429, é só ir mais devagar e continuar de onde parou —
o que já baixou fica em cache:

```bash
./.venv/bin/python -m scripts.build_actors --merge --rate 2
```

### Quando alguém fica de fora

Acontece quando a Wikipédia não tem foto boa da pessoa, quando a foto
principal do artigo não é um retrato, ou quando a rede falhou nas tentativas. Para resolver caso a caso, abra `data/photo_urls.py` e coloque os
links diretos das imagens:

```python
PHOTO_URLS = {
    "Selton Mello": [
        "https://upload.wikimedia.org/.../Selton_Mello.jpg",
        "https://exemplo.com/outra-foto.jpg",
    ],
}
```

O que estiver nesse arquivo tem prioridade sobre a busca automática. Depois
rode com `--merge`, e só quem falhou é refeito:

```bash
./.venv/bin/python -m scripts.build_actors --merge
```

Para trocar quem entra na comparação pela busca automática, edite a lista em
`data/actors.py` (use o título do artigo na Wikipédia em inglês) e rode o build
de novo.

## As suas próprias fotos

Se preferir montar a base na mão — ou ir enchendo aos poucos —, jogue as fotos
na pasta `fotos/` e rode:

```bash
./.venv/bin/python -m scripts.build_actors --photos-dir --merge
```

Os nomes vêm da própria pasta, sem editar código. Dois jeitos, que podem
conviver:

```
fotos/
  Tom Hanks.jpg          <- arquivo solto: o nome do arquivo é o nome da pessoa
  Fernanda Torres/       <- pasta: várias fotos, e o app tira a média dos rostos
    1.jpg
    2.jpg
```

Com `--merge`, cada execução **soma** à base: você joga mais uma foto na pasta,
roda de novo, e quem já estava lá continua. Sem `--merge` a base é refeita do
zero e quem não está na pasta some.

Serve tanto para atores que a Wikipédia não cobre bem quanto para brincar
comparando com amigos e família. Detalhes em `fotos/LEIA-ME.md`.

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
2.2 (o que o facenet-pytorch pede) e com o 2.13 + numpy 2, com resultados
idênticos.

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
| `data/photo_urls.py` | links de fotos definidos na mão (opcional) |
| `fotos/` | suas próprias fotos, uma pasta ou arquivo por pessoa |
| `static/` | página, estilo e JS da câmera |
| `static/share.js` | monta a imagem de resultado no navegador |
| `static/privacidade.html` | página de privacidade |

## Limitações

- A base só compara com os atores que você montou — quem não está na lista nunca
  vai aparecer.
- Rosto de lado, óculos escuros ou luz ruim derrubam a qualidade do resultado. O
  app avisa quando detecta isso, mas ainda assim responde.
- A cartela de cores depende de enxergar o cabelo: se a foto corta o topo da
  cabeça, o app usa só pele e olhos e marca a confiança como mais baixa.
- Uma foto por ator já funciona, mas 4–6 deixam o resultado bem mais estável.
