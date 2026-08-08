# Imagem de produção do Iaface.
#
# Build em dois estágios para a imagem final não carregar compilador nem cache
# do pip. Os pesos do modelo (~110 MB) são baixados durante o build, para o
# contêiner subir sem depender da rede.

FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip install --no-deps facenet-pytorch==2.6.0

# Baixa os pesos agora: sem isso, a primeira requisição em produção pagaria o
# download, e um contêiner sem saída para a internet nem funcionaria.
RUN python -c "from facenet_pytorch import MTCNN, InceptionResnetV1; \
    MTCNN(); InceptionResnetV1(pretrained='vggface2')"


FROM python:3.12-slim

# curl entra por causa do healthcheck; sem ele o orquestrador não sabe se o
# modelo terminou de carregar.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 iaface

COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=iaface:iaface /root/.cache/torch /home/iaface/.cache/torch

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    IAFACE_PUBLIC=1 \
    IAFACE_TRUST_PROXY=1

WORKDIR /app
COPY --chown=iaface:iaface app ./app
COPY --chown=iaface:iaface data ./data
COPY --chown=iaface:iaface static ./static
COPY --chown=iaface:iaface scripts ./scripts

USER iaface
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD curl -fsS http://127.0.0.1:8000/api/status || exit 1

# Um worker por contêiner: o modelo ocupa memória demais para replicar dentro
# do mesmo processo. Para escalar, suba mais contêineres.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]
