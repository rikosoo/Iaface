/**
 * Montagem e envio da imagem de resultado.
 *
 * Tudo acontece no navegador: o card é desenhado num canvas local e só sai
 * daqui quando a pessoa clica em compartilhar. A foto do usuário fica de fora
 * por padrão — quem quiser incluí-la marca a caixa.
 */

const CARD = { w: 1080, h: 1350 };

const CORES = {
  fundo: "#0d0f14",
  painel: "#161a22",
  linha: "#262c38",
  texto: "#eef1f6",
  fraco: "#98a1b2",
  destaque: "#6c8cff",
  destaque2: "#b06cff",
};

let ultimo = null; // { data, faceImg }

/** Texto que acompanha a imagem nas redes que aceitam legenda. */
function legenda(data) {
  const nomes = data.matches.map((m) => `${m.name} (${m.percent}%)`).join(", ");
  const cartela = data.style ? ` Minha cartela de cores: ${data.style.season}.` : "";
  return `Descobri com quais atores eu pareço: ${nomes}.${cartela}`;
}

function arredondado(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function circulo(ctx, img, cx, cy, raio) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, raio, 0, Math.PI * 2);
  ctx.clip();
  const lado = Math.min(img.width, img.height);
  ctx.drawImage(
    img,
    (img.width - lado) / 2,
    (img.height - lado) / 2,
    lado,
    lado,
    cx - raio,
    cy - raio,
    raio * 2,
    raio * 2
  );
  ctx.restore();
}

function carregar(src) {
  return new Promise((resolve) => {
    if (!src) return resolve(null);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

/** Desenha o card completo. Devolve o canvas pronto. */
async function desenhar(data, incluirFoto) {
  const canvas = document.getElementById("card");
  canvas.width = CARD.w;
  canvas.height = CARD.h;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = CORES.fundo;
  ctx.fillRect(0, 0, CARD.w, CARD.h);

  const brilho = ctx.createLinearGradient(0, 0, CARD.w, 0);
  brilho.addColorStop(0, CORES.destaque);
  brilho.addColorStop(1, CORES.destaque2);

  ctx.textAlign = "center";
  ctx.fillStyle = brilho;
  ctx.font = "bold 62px system-ui, sans-serif";
  ctx.fillText("Com quem eu pareço", CARD.w / 2, 110);

  let y = 190;

  if (incluirFoto && ultimo.faceImg) {
    circulo(ctx, ultimo.faceImg, CARD.w / 2, y + 90, 90);
    ctx.strokeStyle = CORES.linha;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(CARD.w / 2, y + 90, 90, 0, Math.PI * 2);
    ctx.stroke();
    y += 215;
  }

  const fotos = await Promise.all(data.matches.map((m) => carregar(m.thumb)));

  for (let i = 0; i < data.matches.length; i++) {
    const m = data.matches[i];
    const alturaCartao = 150;
    const margem = 70;
    const largura = CARD.w - margem * 2;

    ctx.fillStyle = CORES.painel;
    arredondado(ctx, margem, y, largura, alturaCartao, 28);
    ctx.fill();
    ctx.strokeStyle = i === 0 ? CORES.destaque : CORES.linha;
    ctx.lineWidth = i === 0 ? 3 : 2;
    ctx.stroke();

    const centroY = y + alturaCartao / 2;
    if (fotos[i]) {
      circulo(ctx, fotos[i], margem + 85, centroY, 52);
    } else {
      ctx.fillStyle = CORES.linha;
      ctx.beginPath();
      ctx.arc(margem + 85, centroY, 52, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.textAlign = "left";
    ctx.fillStyle = CORES.texto;
    ctx.font = "bold 40px system-ui, sans-serif";
    ctx.fillText(`${i + 1}. ${m.name}`, margem + 165, centroY - 8);

    ctx.fillStyle = CORES.fraco;
    ctx.font = "28px system-ui, sans-serif";
    ctx.fillText(`${m.percent}% — ${m.label}`, margem + 165, centroY + 34);

    y += alturaCartao + 22;
  }

  if (data.style) {
    y += 34;
    ctx.textAlign = "center";
    ctx.fillStyle = CORES.texto;
    ctx.font = "bold 36px system-ui, sans-serif";
    ctx.fillText(data.style.season.split("·")[0].trim(), CARD.w / 2, y);

    y += 46;
    const cores = data.style.palette.slice(0, 8);
    const larguraTotal = CARD.w - 140;
    const passo = larguraTotal / cores.length;
    cores.forEach((c, i) => {
      ctx.fillStyle = c.hex;
      arredondado(ctx, 70 + i * passo + 6, y, passo - 12, 74, 14);
      ctx.fill();
    });
    y += 74;
  }

  ctx.textAlign = "center";
  ctx.fillStyle = CORES.fraco;
  ctx.font = "26px system-ui, sans-serif";
  ctx.fillText("Iaface · roda no seu computador, sem enviar sua foto", CARD.w / 2, CARD.h - 46);

  return canvas;
}

function paraBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

function baixar(blob, nome = "iaface.png") {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = nome;
  link.click();
  URL.revokeObjectURL(url);
}

async function atualizarCard() {
  if (!ultimo) return;
  await desenhar(ultimo.data, document.getElementById("include-photo").checked);
}

/** Chamado quando chega um resultado novo. */
async function prepararCompartilhamento(data) {
  ultimo = { data, faceImg: await carregar(data.face_crop) };

  document.getElementById("include-photo").checked = false;
  await atualizarCard();

  // O botão nativo é o único caminho para Instagram e TikTok: nenhum dos dois
  // aceita publicar por URL. No celular ele abre a bandeja com os dois.
  const podeArquivo =
    navigator.canShare && navigator.canShare({ files: [new File([""], "x.png", { type: "image/png" })] });
  document.getElementById("share-native").hidden = !podeArquivo;
  document.getElementById("share-hint").textContent = podeArquivo
    ? "Compartilhar… abre a bandeja do sistema, com Instagram, TikTok e WhatsApp."
    : "Instagram e TikTok não aceitam publicar por link: baixe a imagem e suba por lá. Pelo celular aparece o botão de compartilhar direto.";

  document.getElementById("share").hidden = false;
}

async function compartilharNativo() {
  const canvas = document.getElementById("card");
  const blob = await paraBlob(canvas);
  const file = new File([blob], "iaface.png", { type: "image/png" });
  try {
    await navigator.share({ files: [file], text: legenda(ultimo.data) });
  } catch (err) {
    if (err.name !== "AbortError") baixar(blob);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("include-photo").addEventListener("change", atualizarCard);

  document.getElementById("share-native").addEventListener("click", compartilharNativo);

  document.getElementById("share-download").addEventListener("click", async () => {
    baixar(await paraBlob(document.getElementById("card")));
  });

  document.getElementById("share-whatsapp").addEventListener("click", async () => {
    // O WhatsApp web aceita texto por link, mas não imagem: baixamos o card
    // para a pessoa anexar na conversa que ela escolher.
    baixar(await paraBlob(document.getElementById("card")));
    window.open(`https://wa.me/?text=${encodeURIComponent(legenda(ultimo.data))}`, "_blank", "noopener");
  });

  document.getElementById("share-pinterest").addEventListener("click", async () => {
    // O Pinterest só fixa imagem que já esteja publicada numa URL, e a nossa
    // existe apenas neste navegador — então baixamos e abrimos a criação.
    baixar(await paraBlob(document.getElementById("card")));
    window.open(
      `https://www.pinterest.com/pin-builder/?description=${encodeURIComponent(legenda(ultimo.data))}`,
      "_blank",
      "noopener"
    );
  });

  document.getElementById("share-copy").addEventListener("click", async (e) => {
    await navigator.clipboard.writeText(legenda(ultimo.data));
    const botao = e.target;
    botao.textContent = "Copiado!";
    setTimeout(() => (botao.textContent = "Copiar texto"), 1500);
  });
});

window.prepararCompartilhamento = prepararCompartilhamento;
