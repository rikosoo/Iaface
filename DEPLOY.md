# Publicar o Iaface no seu site

Guia do modo público: o app deixa de rodar na máquina de quem usa e passa a
rodar no **seu** servidor. Isso muda o que você precisa fazer no código e o que
você passa a dever legalmente.

---

## Antes de qualquer coisa: leia esta parte

Em `localhost`, a foto nunca sai do computador de quem tira. Publicado, ela
viaja até o seu servidor — e **você vira o controlador dos dados**.

Foto de rosto analisada para medir semelhança é dado biométrico. No GDPR isso
cai no **Artigo 9**, onde o tratamento é *proibido por padrão*, salvo exceção.
A exceção que se aplica aqui é o **consentimento explícito** (Art. 9(2)(a)).

O que o código já entrega:

- consentimento explícito exigido antes da câmera, com um cabeçalho que o
  servidor confere (`IAFACE_PUBLIC=1`)
- a foto nunca toca o disco, nem em arquivo temporário
- o vetor facial não sobrevive à requisição
- log sem corpo de requisição
- página de privacidade em `/privacidade`

O que **só você** pode providenciar:

- **Registro das atividades de tratamento** (Art. 30)
- **DPIA** (Art. 35) — tratamento de biometria em larga escala praticamente
  sempre exige
- **Aviso de privacidade próprio**, com você identificado como controlador,
  base legal, prazo de retenção e direitos do titular
- **Representante na UE** (Art. 27), se você não tem estabelecimento lá mas
  atende usuários de lá
- **Idade mínima** — consentimento de menor tem regra própria (Art. 8)

E um ponto para checar com um advogado antes de publicar: a análise de cores
**infere tom de pele**. Isso conversa com o conceito de *categorização
biométrica* no AI Act europeu, que restringe sistemas que deduzem
características de pessoas a partir de dados biométricos. Se essa conversa for
desconfortável, dá para desligar a cartela e manter só a comparação com atores.

> Nada aqui é aconselhamento jurídico — é a lista do que costuma ser exigido,
> para você levar a quem entende do assunto.

---

## Subindo

Você precisa de um servidor com Docker, **2 GB de RAM** e um subdomínio
apontado para ele.

```bash
git clone <este-repo> && cd Iaface

# 1. monte a base de atores (roda uma vez, na sua máquina ou no servidor)
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install --no-deps facenet-pytorch==2.6.0
./.venv/bin/python -m scripts.build_actors

# 2. troque o domínio e o e-mail
nano Caddyfile

# 3. no ar
docker compose up -d
```

O Caddy emite o certificado TLS sozinho. **HTTPS não é enfeite**: o navegador
só libera a câmera em contexto seguro, então sem certificado o app não funciona
fora de `localhost`.

A base (`data/actors.npz`) e as miniaturas (`static/actors/`) entram na imagem
no build — gere antes de rodar o `docker compose`.

---

## Embutindo no seu site

### Opção A — iframe (recomendada)

```html
<iframe
  src="https://iaface.seusite.com"
  allow="camera"
  style="width:100%;height:1000px;border:0;border-radius:16px"
  title="Com qual ator você se parece?">
</iframe>
```

O `allow="camera"` é obrigatório: sem ele o navegador bloqueia a câmera dentro
do iframe, sem nem perguntar. E o `frame-ancestors` do `Caddyfile` precisa
listar o domínio do seu site, senão o iframe é recusado.

Vantagem: a página, o consentimento e a política de privacidade vêm prontos e
consistentes entre si.

### Opção B — só a API

Se você quer o frontend com a sua identidade visual:

```js
const resposta = await fetch("https://iaface.seusite.com/api/match", {
  method: "POST",
  headers: {
    "Content-Type": "image/jpeg",
    "X-Iaface-Consent": "granted",   // só depois do opt-in explícito
  },
  body: blobDaFoto,                  // corpo cru, não FormData
});
const resultado = await resposta.json();
```

Libere seu domínio no CORS:

```yaml
IAFACE_CORS_ORIGINS: "https://www.seusite.com"
```

Nesta opção **o aviso de privacidade e o consentimento passam a ser sua
responsabilidade** — você está trocando a interface que os trazia embutidos.

Respostas que o seu front precisa tratar:

| Código | O que aconteceu |
| --- | --- |
| `200` | Deu certo |
| `403` | Faltou o cabeçalho de consentimento |
| `413` | Imagem acima de 12 MB |
| `422` | Nenhum rosto encontrado na foto |
| `429` | Limite por IP estourado — respeite o `Retry-After` |
| `503` | Fila cheia; tente de novo em alguns segundos |

---

## Ajustes de carga

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `IAFACE_PUBLIC` | `0` | Liga o consentimento obrigatório e os limites |
| `IAFACE_MAX_CONCURRENCY` | `2` | Análises simultâneas (~200 MB cada) |
| `IAFACE_QUEUE_TIMEOUT` | `20` | Espera na fila antes de responder 503 |
| `IAFACE_RATE_LIMIT` | `10` | Análises por IP na janela |
| `IAFACE_RATE_WINDOW` | `60` | Tamanho da janela, em segundos |
| `IAFACE_CORS_ORIGINS` | vazio | Domínios liberados (só na opção B) |
| `IAFACE_TRUST_PROXY` | `0` | Ler `X-Forwarded-For`. **Ligue só com proxy na frente** |

Sobre o `TRUST_PROXY`: o cabeçalho é trivial de forjar. Ligado sem um proxy
confiável na frente, qualquer um inventa um IP novo a cada requisição e o
limite por IP vira decoração.

**Dimensionamento.** Cada análise custa ~0,2–2s de CPU e ~200 MB. Uma VPS de
2 vCPU e 2 GB atende algo como 1–2 análises por segundo. Para mais que isso,
suba mais contêineres atrás do Caddy — não aumente `MAX_CONCURRENCY` além do
que a RAM aguenta, porque o resultado é o processo morrer por falta de memória
em vez de enfileirar.

---

## O que foi medido

A inferência é síncrona. Rodando direto no endpoint `async`, ela travava o laço
de eventos e, com ele, todas as outras requisições:

| | Antes | Depois |
| --- | --- | --- |
| `/api/status` ocioso | 2ms | 2ms |
| `/api/status` durante 3 análises | **776ms** | **3ms** |

A correção foi mandar o trabalho pesado para um thread pool com um teto de
concorrência. Sem ela, cinco pessoas ao mesmo tempo engasgam a página inteira —
inclusive os arquivos estáticos.

---

## Checklist antes de anunciar

- [ ] `IAFACE_PUBLIC=1` está de fato ligado (confira em `/api/status`)
- [ ] HTTPS funcionando, e a câmera abrindo pelo domínio real
- [ ] `frame-ancestors` no `Caddyfile` lista o domínio do seu site
- [ ] O consentimento aparece **antes** do botão da câmera
- [ ] `/privacidade` traz você como controlador, e não o texto de exemplo
- [ ] DPIA feita e registro de tratamento preenchido
- [ ] Testado no celular — é de onde vem a maioria dos acessos
- [ ] Limite por IP testado: 11 análises seguidas devem dar 429
