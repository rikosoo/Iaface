const $ = (id) => document.getElementById(id);

const video = $("video");
const canvas = $("canvas");
const shot = $("shot");
const overlay = $("overlay");
const msg = $("msg");

let stream = null;

/** Escapa texto vindo do servidor antes de injetar no HTML. */
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => `&${{ "&": "amp", "<": "lt", ">": "gt", '"': "quot", "'": "#39" }[c]};`);

function say(text, isError = false) {
  msg.textContent = text;
  msg.classList.toggle("error", isError);
}

function hideResults() {
  ["verdict", "results", "style", "warnings", "share"].forEach((id) => ($(id).hidden = true));
}

// --- Câmera ---------------------------------------------------------------

async function startCamera() {
  say("");
  hideResults();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
  } catch (err) {
    overlay.innerHTML = "<p>Câmera bloqueada</p>";
    overlay.hidden = false;
    say(
      err.name === "NotAllowedError"
        ? "Você precisa permitir o acesso à câmera no navegador."
        : `Não consegui abrir a câmera (${err.name}). Use "Enviar arquivo".`,
      true
    );
    return;
  }

  video.srcObject = stream;
  video.hidden = false;
  shot.hidden = true;
  overlay.hidden = true;
  $("start").hidden = true;
  $("capture").hidden = false;
  $("retry").hidden = true;
  say("Enquadre o rosto, com luz de frente, e tire a foto.");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

function captureBlob() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

async function capture() {
  if (!video.videoWidth) return say("A câmera ainda está carregando…", true);

  const blob = await captureBlob();
  showPreview(URL.createObjectURL(blob), true);
  stopCamera();

  $("capture").hidden = true;
  $("retry").hidden = false;
  await send(blob);
}

function showPreview(src, mirrored) {
  shot.src = src;
  shot.classList.toggle("mirrored", mirrored);
  shot.hidden = false;
  video.hidden = true;
  overlay.hidden = true;
}

// --- Envio ----------------------------------------------------------------

async function send(blob) {
  say("Analisando seu rosto…");
  hideResults();

  const buttons = document.querySelectorAll("button, .upload");
  buttons.forEach((b) => b.classList.add("busy"));
  document.querySelectorAll("button").forEach((b) => (b.disabled = true));

  try {
    // Corpo cru, e não FormData: o parser multipart do servidor derramaria
    // a foto para um arquivo temporário em disco.
    const res = await fetch("/api/match", {
      method: "POST",
      headers: { "Content-Type": blob.type || "image/jpeg" },
      body: blob,
    });
    const data = await res.json();

    if (!res.ok) {
      say(data.detail || "Algo deu errado na análise.", true);
      return;
    }
    render(data);
    say("");
  } catch (err) {
    say(`Falha ao falar com o servidor: ${err.message}`, true);
  } finally {
    buttons.forEach((b) => b.classList.remove("busy"));
    document.querySelectorAll("button").forEach((b) => (b.disabled = false));
  }
}

// --- Resultado ------------------------------------------------------------

function render(data) {
  renderWarnings(data.quality);
  renderVerdict(data);
  renderMatches(data.matches);
  renderStyle(data.style);
  window.prepararCompartilhamento(data);
  $("verdict").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderWarnings(quality) {
  const list = $("warnings");
  if (!quality.warnings.length) return;
  list.innerHTML = quality.warnings.map((w) => `<li>${esc(w)}</li>`).join("");
  list.hidden = false;
}

function renderVerdict(data) {
  $("analyzed").src = data.face_crop;
  $("verdict-text").textContent = data.verdict;
  $("verdict-sub").textContent =
    "Este é o recorte que o modelo analisou — ele foi comparado com cada ator da base.";
  $("verdict").hidden = false;
}

function renderMatches(matches) {
  $("results").innerHTML = matches
    .map(
      (m, i) => `
      <article class="card">
        <span class="rank">#${i + 1}</span>
        ${
          m.thumb
            ? `<img src="${esc(m.thumb)}" alt="${esc(m.name)}" loading="lazy" />`
            : `<div class="noimg">🎬</div>`
        }
        <h3>${esc(m.name)}</h3>
        <div class="bar"><i style="width:${m.percent}%"></i></div>
        <p class="pct">${m.percent}% — ${esc(m.label)}</p>
        <p class="raw">cosseno ${m.similarity.toFixed(3)}</p>
      </article>`
    )
    .join("");
  $("results").hidden = false;
}

function swatches(colors) {
  return colors
    .map(
      (c) => `
      <figure class="swatch">
        <span style="background:${esc(c.hex)}"></span>
        <figcaption>${esc(c.name)}</figcaption>
      </figure>`
    )
    .join("");
}

function renderStyle(style) {
  const section = $("style");
  if (!style) {
    section.hidden = true;
    return;
  }

  $("style-season").textContent = `${style.season} — ${style.idea}`;

  const tones = [
    ["Pele", style.skin_hex, `subtom ${style.undertone}, profundidade ${style.depth}`],
    ["Cabelo", style.hair_hex, `contraste ${style.contrast}`],
    ["Olhos", style.eyes_hex, ""],
  ].filter(([, hex]) => hex);

  $("tones").innerHTML = tones
    .map(
      ([label, hex, note]) => `
      <div class="tone">
        <span class="dot" style="background:${esc(hex)}"></span>
        <div>
          <strong>${esc(label)}</strong>
          <small>${esc(hex)}${note ? " · " + esc(note) : ""}</small>
        </div>
      </div>`
    )
    .join("");

  $("palette").innerHTML = swatches(style.palette);
  $("avoid").innerHTML = swatches(style.avoid);
  $("pieces").innerHTML = [...style.pieces, `Metais: ${style.metals}`, style.contrast_tip]
    .map((p) => `<li>${esc(p)}</li>`)
    .join("");

  const notes = [...style.notes];
  notes.push(`Confiança da leitura de cor: ${style.confidence}.`);
  $("style-notes").textContent = notes.join(" ");
  section.hidden = false;
}

// --- Ligações -------------------------------------------------------------

$("start").addEventListener("click", startCamera);
$("capture").addEventListener("click", capture);
$("retry").addEventListener("click", startCamera);
$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopCamera();
  showPreview(URL.createObjectURL(file), false);
  $("capture").hidden = true;
  $("start").hidden = false;
  await send(file);
  e.target.value = "";
});

fetch("/api/status")
  .then((r) => r.json())
  .then((s) => {
    $("dbinfo").textContent = s.ready
      ? `${s.actors} atores na base`
      : "Base de atores vazia — rode: python -m scripts.build_actors";
    if (!s.ready) say("A base de atores ainda não foi gerada.", true);
  })
  .catch(() => {});
