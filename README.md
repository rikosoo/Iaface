# Iaface — com qual ator você se parece?

App que roda no seu computador: abre a câmera no navegador, tira uma foto e
mostra o **top 3 de atores** mais parecidos com você.

Nenhuma foto sai da sua máquina — a câmera, o servidor e o modelo rodam todos
em `localhost`.

## Como funciona

1. **MTCNN** encontra e alinha o rosto na foto.
2. A rede **InceptionResnetV1 treinada no VGGFace2** transforma o rosto em um
   vetor de 512 números (o "embedding" do rosto).
3. Esse vetor é comparado por **similaridade de cosseno** com a média dos
   embeddings de cada ator da base, e os 3 maiores viram o resultado.

O modelo não foi treinado para dizer "você é o fulano" — ele mede proximidade
entre rostos. É uma brincadeira, não uma identificação.

## Instalação

Precisa de Python 3.9+ (testado no 3.11).

```bash
git clone <este-repo> && cd Iaface
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

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
--limit 20        # só os 20 primeiros, pra testar rápido
--per-actor 6     # mais fotos por ator = comparação mais estável
--photos-dir fotos  # usa fotos suas em vez de baixar da internet
```

No modo `--photos-dir`, organize assim:

```
fotos/
  Tom Hanks/1.jpg
  Tom Hanks/2.jpg
  Fernanda Torres/1.jpg
```

Para trocar quem entra na comparação, edite a lista em `data/actors.py`
(use o título do artigo na Wikipédia em inglês) e rode o build de novo.

## 2) Rodar

```bash
./run.sh
```

ou, na mão:

```bash
./.venv/bin/uvicorn app.server:app --port 8000
```

Abra **http://localhost:8000**, clique em *Abrir câmera*, permita o acesso e
tire a foto. Também dá para enviar uma imagem do disco.

> A câmera só funciona em `localhost` ou HTTPS — é uma regra do navegador. Se
> você abrir pelo IP da máquina (`http://192.168...`), o botão da câmera falha;
> nesse caso use o envio de arquivo ou um túnel HTTPS.

## Estrutura

| Arquivo | O que faz |
| --- | --- |
| `app/face.py` | detecção do rosto e geração do embedding |
| `app/server.py` | API FastAPI (`/api/match`, `/api/status`) e arquivos estáticos |
| `scripts/build_actors.py` | baixa fotos, calcula embeddings e grava a base |
| `data/actors.py` | lista de atores de referência |
| `static/` | página, estilo e JS da câmera |

## Sobre a porcentagem

A similaridade de cosseno útil fica entre ~0,2 e ~0,9; o servidor estica essa
faixa para 0–100% só para o número ficar legível. Duas fotos da mesma pessoa
passam de 80%; "parecidos" costumam ficar entre 40% e 60%.

## Limitações

- A base só compara com os atores que você montou — quem não está na lista
  nunca vai aparecer.
- Iluminação ruim, óculos escuros, rosto de lado ou muito pequeno na imagem
  derrubam a qualidade do resultado.
- Uma foto por ator já funciona, mas 4–6 fotos deixam o resultado bem mais
  estável.
