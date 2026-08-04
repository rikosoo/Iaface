const $ = (id) => document.getElementById(id);

const video = $("video");
const canvas = $("canvas");
const shot = $("shot");
const overlay = $("overlay");
const msg = $("msg");
const results = $("results");

let stream = null;

function say(text, isError = false) {
  msg.textContent = text;
  msg.classList.toggle("error", isError);
}

function showOverlay(html) {
  overlay.innerHTML = html;
  overlay.hidden = false;
}

async function startCamera() {
  say("");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
  } catch (err) {
    showOverlay("<p>Câmera bloqueada</p>");
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
  results.hidden = true;
  say("Enquadre o rosto e tire a foto.");
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
  shot.src = URL.createObjectURL(blob);
  shot.classList.add("mirrored");
  shot.hidden = false;
  video.hidden = true;
  stopCamera();

  $("capture").hidden = true;
  $("retry").hidden = false;
  await send(blob);
}

async function send(blob) {
  say("Analisando seu rosto…");
  results.hidden = true;
  const buttons = document.querySelectorAll("button");
  buttons.forEach((b) => (b.disabled = true));

  try {
    const form = new FormData();
    form.append("photo", blob, "foto.jpg");
    const res = await fetch("/api/match", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      say(data.detail || "Algo deu errado na análise.", true);
      return;
    }
    render(data.matches);
    say("");
  } catch (err) {
    say(`Falha ao falar com o servidor: ${err.message}`, true);
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

function render(matches) {
  results.innerHTML = matches
    .map(
      (m, i) => `
      <article class="card">
        <span class="rank">#${i + 1}</span>
        ${
          m.thumb
            ? `<img src="${m.thumb}" alt="${m.name}" loading="lazy" />`
            : `<div class="noimg">🎬</div>`
        }
        <h3>${m.name}</h3>
        <div class="bar"><i style="width:${m.percent}%"></i></div>
        <p class="pct">${m.percent}% de semelhança</p>
      </article>`
    )
    .join("");
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("start").addEventListener("click", startCamera);
$("capture").addEventListener("click", capture);
$("retry").addEventListener("click", startCamera);
$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopCamera();
  video.hidden = true;
  overlay.hidden = true;
  shot.src = URL.createObjectURL(file);
  shot.classList.remove("mirrored");
  shot.hidden = false;
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
