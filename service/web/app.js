/* ReID · веб-интерфейс оператора. Работает только с API сервиса (/api/*), без внешних ресурсов. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const PAD = 0.05;                              // запас вокруг рамки — как в модели (reid.data.crop)
  const S = {
    img: null, blob: null, W: 0, H: 0, bbox: null, dets: [], sampleId: null,
    result: null, active: 0, explain: new Map(), readonly: false, canExplain: false,
    galOffset: 0, galLimit: 60, queryCrop: null, queryAspect: 1, token: 0, drawMode: false,
  };

  // ---------------------------------------------------------------- утилиты
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch { /* не JSON */ }
    if (!res.ok) throw new Error((data && data.detail) || `ошибка ${res.status}`);
    return data;
  }
  function toast(text, kind = "") {
    const t = document.createElement("div");
    t.className = `toast ${kind}`;
    t.textContent = text;
    $("#toasts").appendChild(t);
    setTimeout(() => t.remove(), 4200);
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  // COCO путает легковые и грузовые (седан бывает «грузовиком») — для них показываем просто «машина»
  const kindText = k => (k === "автобус" || k === "мотоцикл" ? k : "машина");
  const band = (s, thr) => (s >= 0.6 ? "hi" : s >= thr ? "mid" : "lo");
  const bandName = { hi: "высокая", mid: "средняя", lo: "ниже порога" };
  const camText = c => (c == null ? "кам. —" : `кам. ${c}`);
  function download(name, text, type) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  function confirmBox(title, text) {
    return new Promise(done => {
      $("#confirmTitle").textContent = title;
      $("#confirmText").textContent = text;
      $("#confirm").classList.remove("hidden");
      const close = v => { $("#confirm").classList.add("hidden"); done(v); };
      $("#confirmYes").onclick = () => close(true);
      $("#confirmNo").onclick = () => close(false);
    });
  }

  // ---------------------------------------------------------------- тема и вкладки
  try { const t = localStorage.getItem("reid_theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* нет хранилища */ }
  $("#themeBtn").onclick = () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("reid_theme", t); } catch { /* нет хранилища */ }
  };
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
    document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-selected", x === b));
    document.querySelectorAll(".tab-page").forEach(p => p.classList.toggle("hidden", p.id !== `tab-${b.dataset.tab}`));
    if (b.dataset.tab === "gallery" && !$("#galGrid").children.length) loadGallery(true);
  });

  // ---------------------------------------------------------------- состояние сервиса
  async function health() {
    try {
      const h = await api("/api/health");
      $("#stState").className = "chip ok";
      $("#stState span").textContent = "сервис готов";
      $("#stGallery").textContent = `${h.gallery.toLocaleString("ru")} ТС в базе`;
      $("#stDevice").textContent = (h.providers[0] || "").startsWith("CUDA") ? "GPU" : "CPU";
      $("#stReadonly").classList.toggle("hidden", !h.readonly);
      S.readonly = h.readonly;
      S.canExplain = h.explain;
      $("#heatSwitch").classList.toggle("hidden", !h.explain);
      $("#galLoadDemo").disabled = $("#galClear").disabled = h.readonly;
      $("#galInfo").textContent = `${h.gallery.toLocaleString("ru")} объектов · индекс ${h.index} · ${h.dim} чисел на машину · порог ${h.threshold}` +
        (h.readonly ? " · демо-режим: база только для чтения" : "");
      $("#galTitle").textContent = `Галерея · ${h.gallery.toLocaleString("ru")} ТС`;
      return h;
    } catch {
      $("#stState").className = "chip bad";
      $("#stState span").textContent = "сервис недоступен";
      return null;
    }
  }

  // ---------------------------------------------------------------- кадр и рамки
  const frame = $("#frame"), svg = $("#overlay");
  const NS = "http://www.w3.org/2000/svg";

  async function setFrame(blob, preset, sampleId = null) {
    const token = ++S.token;
    S.blob = blob; S.sampleId = sampleId; S.dets = []; S.bbox = preset || null;
    S.explain.clear(); setDrawMode(false); renderDets();
    const url = URL.createObjectURL(blob);
    await new Promise((ok, fail) => { frame.onload = ok; frame.onerror = fail; frame.src = url; })
      .catch(() => { throw new Error("не удалось открыть изображение"); });
    if (token !== S.token) return;
    S.img = frame; S.W = frame.naturalWidth; S.H = frame.naturalHeight;
    frame.hidden = false; svg.hidden = false; $("#stageEmpty").classList.add("hidden");
    svg.setAttribute("viewBox", `0 0 ${S.W} ${S.H}`);
    $("#wholeBtn").disabled = $("#drawBtn").disabled = false;
    updateCrop(); draw(); readyState();
    $("#stageInfo").textContent = "Ищем машины на кадре…";
    $("#stageBusy").classList.remove("hidden");
    try {
      const f = new FormData();
      f.append("file", blob, "frame.jpg");
      const d = await api("/api/detect", { method: "POST", body: f });
      if (token !== S.token) return;
      S.dets = d.vehicles || [];
      if (!S.bbox && S.dets.length) S.bbox = { ...S.dets[0] };
      info(d);
    } catch (e) {
      if (token === S.token) $("#stageInfo").textContent = `Автоопределение недоступно (${e.message}) — нажмите «Своя рамка» и обведите машину.`;
    } finally {
      if (token === S.token) $("#stageBusy").classList.add("hidden");
    }
    updateCrop(); draw(); readyState(); renderDets();
  }

  function info(d) {
    const n = S.dets.length;
    const box = $("#stageInfo");
    box.textContent = "";
    if (!n) { box.textContent = "Машины не найдены автоматически — нажмите «Своя рамка» и обведите машину."; return; }
    box.append("Найдено машин: ", el("b", null, String(n)), ` за ${Math.round(d.ms)} мс. ` +
      (n > 1 ? "Выберите нужную — ниже или прямо на кадре." : "Если рамка не та — «Своя рамка»."));
  }

  // найденные машины — крупные кнопки-превью: на телефоне в них легко попасть пальцем
  function renderDets() {
    const box = $("#dets");
    box.textContent = "";
    box.classList.toggle("hidden", !S.dets.length);
    S.dets.forEach((d, i) => {
      const b = el("button", "det-chip");
      b.type = "button";
      const cv = document.createElement("canvas");
      const k = Math.min(1, 200 / Math.max(d.w, d.h));
      cv.width = Math.max(1, Math.round(d.w * k)); cv.height = Math.max(1, Math.round(d.h * k));
      try { cv.getContext("2d").drawImage(S.img, d.x, d.y, d.w, d.h, 0, 0, cv.width, cv.height); } catch { /* кадр ещё не готов */ }
      const th = el("span", "det-thumb"); th.appendChild(cv);
      b.append(th, el("span", "det-cap", `${i + 1} · ${kindText(d.kind)} ${Math.round(d.score * 100)}%`));
      b.onclick = () => choose({ ...d });
      box.appendChild(b);
    });
    syncDets();
  }
  function syncDets() {
    document.querySelectorAll("#dets .det-chip").forEach((b, i) =>
      b.classList.toggle("active", !!S.bbox && iou(S.dets[i], S.bbox) > 0.6));
  }
  function choose(b) {
    S.bbox = b && { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.w), h: Math.round(b.h) };
    updateCrop(); draw(); readyState(); syncDets();
  }

  // режим «своя рамка»: мышью рисовать можно всегда, на сенсорном экране — только в этом режиме,
  // иначе палец над кадром не даёт прокручивать страницу
  function setDrawMode(on) {
    S.drawMode = on;
    $("#drawBtn").setAttribute("aria-pressed", on);
    $("#stage").classList.toggle("drawing", on);
  }
  $("#drawBtn").onclick = () => {
    setDrawMode(!S.drawMode);
    if (S.drawMode) toast("Обведите машину на кадре — мышью или пальцем");
  };

  function draw() {
    svg.textContent = "";
    if (!S.W) return;
    const sw = Math.max(2, S.W / 420);
    if (S.bbox) {                               // затемнение всего, кроме выбранной машины
      const { x, y, w, h } = S.bbox;
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", `M0 0H${S.W}V${S.H}H0Z M${x} ${y}V${y + h}H${x + w}V${y}Z`);
      p.setAttribute("fill-rule", "evenodd");
      p.setAttribute("class", "dim");
      svg.appendChild(p);
    }
    S.dets.forEach((d, i) => {
      if (S.bbox && iou(d, S.bbox) > 0.5) return;
      const r = rect(d, "det", sw);
      r.dataset.i = i;
      svg.appendChild(r);
      const lh = Math.max(26, S.H / 34), fs = lh * 0.68;
      const txt = `${i + 1} · ${kindText(d.kind)} ${Math.round(d.score * 100)}%`;
      const lw = txt.length * fs * 0.56 + fs, lx = Math.max(0, Math.min(d.x, S.W - lw));   // подпись не уезжает за край
      const lab = document.createElementNS(NS, "rect");
      lab.setAttribute("x", lx); lab.setAttribute("y", Math.max(0, d.y - lh));
      lab.setAttribute("width", lw); lab.setAttribute("height", lh);
      lab.setAttribute("rx", 4); lab.setAttribute("class", "det-label");
      svg.appendChild(lab);
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", lx + fs * 0.5); t.setAttribute("y", Math.max(0, d.y - lh) + lh * 0.72);
      t.setAttribute("class", "det-text"); t.style.fontSize = `${fs}px`;
      t.textContent = txt;
      svg.appendChild(t);
    });
    if (S.bbox) {
      svg.appendChild(rect(S.bbox, "sel", sw * 1.6));
      const c = Math.max(10, S.W / 110), { x, y, w, h } = S.bbox;
      [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([cx, cy]) => {
        const k = document.createElementNS(NS, "rect");
        k.setAttribute("x", cx - c / 2); k.setAttribute("y", cy - c / 2);
        k.setAttribute("width", c); k.setAttribute("height", c); k.setAttribute("rx", c / 4);
        k.setAttribute("class", "sel-corner");
        svg.appendChild(k);
      });
    }
  }
  function rect(b, cls, sw) {
    const r = document.createElementNS(NS, "rect");
    r.setAttribute("x", b.x); r.setAttribute("y", b.y); r.setAttribute("width", b.w); r.setAttribute("height", b.h);
    r.setAttribute("class", cls); r.setAttribute("stroke-width", sw);
    return r;
  }
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    return inter / (a.w * a.h + b.w * b.h - inter || 1);
  }
  function pt(ev) {                                // экран -> пиксели исходного кадра
    const p = svg.createSVGPoint();
    p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(svg.getScreenCTM().inverse());
    return { x: Math.min(S.W, Math.max(0, q.x)), y: Math.min(S.H, Math.max(0, q.y)) };
  }

  let drag = null;
  svg.addEventListener("pointerdown", ev => {
    if (!S.W) return;
    const canDraw = ev.pointerType === "mouse" || S.drawMode;
    if (canDraw) svg.setPointerCapture(ev.pointerId);
    drag = { start: pt(ev), moved: false, prev: S.bbox, canDraw };
  });
  svg.addEventListener("pointercancel", () => { if (drag) { S.bbox = drag.prev; drag = null; draw(); } });
  svg.addEventListener("pointermove", ev => {
    if (!drag || !drag.canDraw) return;
    const p = pt(ev), s = drag.start;
    const minMove = S.W / 150;
    if (!drag.moved && Math.hypot(p.x - s.x, p.y - s.y) < minMove) return;
    drag.moved = true;
    S.bbox = { x: Math.min(s.x, p.x), y: Math.min(s.y, p.y), w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) };
    draw();
  });
  svg.addEventListener("pointerup", ev => {
    if (!drag) return;
    const d = drag; drag = null;
    if (!d.moved) {                               // клик/тап: выбрать найденную машину под пальцем
      const p = pt(ev), tol = S.W / 60;           // небольшой допуск — по мелкой рамке легко промахнуться
      const hit = S.dets.filter(b => p.x >= b.x - tol && p.x <= b.x + b.w + tol && p.y >= b.y - tol && p.y <= b.y + b.h + tol)
        .sort((a, b) => a.w * a.h - b.w * b.h)[0];
      if (hit) choose(hit);
      return;
    }
    if (S.bbox.w < 12 || S.bbox.h < 12) {
      S.bbox = d.prev;
      toast("Рамка слишком маленькая — нарисуйте крупнее", "error");
    } else setDrawMode(false);
    choose(S.bbox);
  });

  $("#wholeBtn").onclick = () => choose({ x: 0, y: 0, w: S.W, h: S.H });
  $("#fileInput").onchange = e => { const f = e.target.files[0]; if (f) openFile(f); e.target.value = ""; };
  const stage = $("#stage");
  ["dragenter", "dragover"].forEach(n => stage.addEventListener(n, e => { e.preventDefault(); stage.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(n => stage.addEventListener(n, e => { e.preventDefault(); stage.classList.remove("drag"); }));
  stage.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) openFile(f); });
  async function openFile(f) {
    if (f.type && !f.type.startsWith("image/")) { toast("Нужна фотография (JPEG, PNG)", "error"); return; }
    document.querySelectorAll(".sample").forEach(x => x.classList.remove("active"));
    try { await setFrame(await normalize(f), null); } catch (e) { toast(e.message, "error"); }
  }
  // фото с телефона: поворачиваем по EXIF и уменьшаем до 1920 px — сервер получает ровно то, что видно
  // на экране (иначе рамки детектора не совпадут с кадром), а загрузка идёт в разы быстрее
  async function normalize(f) {
    // через <img>: все браузеры (включая Safari на iPhone) рисуют его на canvas уже с поворотом из EXIF;
    // createImageBitmap в части браузеров поворот игнорирует — фото уходило боком
    const url = URL.createObjectURL(f);
    try {
      const im = new Image();
      im.decoding = "async";
      await new Promise((ok, fail) => { im.onload = ok; im.onerror = fail; im.src = url; });
      const k = Math.min(1, 1920 / Math.max(im.naturalWidth, im.naturalHeight));
      const c = document.createElement("canvas");
      c.width = Math.round(im.naturalWidth * k); c.height = Math.round(im.naturalHeight * k);
      c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
      return await new Promise((ok, fail) => c.toBlob(b => (b ? ok(b) : fail()), "image/jpeg", 0.92));
    } catch {
      return f;                                  // браузер не смог — отправляем как есть, сервер повернёт сам
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // кроп запроса для сравнения — ровно та область, что уходит в модель (рамка + 5%)
  function updateCrop() {
    S.queryCrop = null;
    if (!S.bbox || !S.img) return;
    const { x, y, w, h } = S.bbox;
    const x0 = Math.max(0, x - w * PAD), y0 = Math.max(0, y - h * PAD);
    const x1 = Math.min(S.W, x + w + w * PAD), y1 = Math.min(S.H, y + h + h * PAD);
    const c = document.createElement("canvas");
    const k = Math.min(1, 640 / Math.max(x1 - x0, y1 - y0));
    c.width = Math.max(1, Math.round((x1 - x0) * k)); c.height = Math.max(1, Math.round((y1 - y0) * k));
    c.getContext("2d").drawImage(S.img, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
    S.queryCrop = c.toDataURL("image/jpeg", 0.9);
    S.queryAspect = (x1 - x0) / (y1 - y0);
  }
  function readyState() { $("#goBtn").disabled = !(S.blob && S.bbox); }

  // ---------------------------------------------------------------- примеры
  async function loadSamples() {
    let list = [];
    try { list = await api("/api/demo/queries?limit=24"); } catch { /* демо-данные не смонтированы */ }
    const box = $("#samples");
    box.textContent = "";
    $("#samplesCount").textContent = list.length ? `${list.length} кадров` : "демо-данные не подключены";
    list.forEach((q, i) => {
      const b = el("button", "sample");
      b.type = "button";
      b.setAttribute("aria-label", `пример ${i + 1}`);
      const W = q.width || 1920, H = q.height || 1080;
      b.style.aspectRatio = `${W} / ${H}`;
      const im = document.createElement("img");       // уменьшенный кадр (~30 КБ) + рамка машины-запроса
      im.loading = "lazy"; im.decoding = "async"; im.alt = "";
      im.onload = () => b.classList.add("loaded");
      im.src = `${q.url}?w=360`;
      const bx = el("span", "sample-box");
      Object.assign(bx.style, { left: `${q.x / W * 100}%`, top: `${q.y / H * 100}%`, width: `${q.w / W * 100}%`, height: `${q.h / H * 100}%` });
      b.append(im, bx);
      box.appendChild(b);
      b.onclick = async () => {
        document.querySelectorAll(".sample").forEach(x => x.classList.toggle("active", x === b));
        b.classList.add("busy");
        try { await setFrame(await fetch(q.url).then(r => r.blob()), { x: q.x, y: q.y, w: q.w, h: q.h }, q.image_id); }
        catch (e) { toast(e.message, "error"); }
        finally { b.classList.remove("busy"); }
        if (innerWidth < 1080) $("#stage").scrollIntoView({ behavior: "smooth", block: "start" });
      };
    });
    if (list.length && location.hash === "#demo") {
      await box.firstChild.onclick();
      search();
    }
  }

  // ---------------------------------------------------------------- поиск
  const thr = $("#thr"), topk = $("#topk");
  thr.oninput = () => { $("#thrVal").textContent = (+thr.value).toFixed(2); };
  topk.oninput = () => { $("#topkVal").textContent = topk.value; };
  $("#goBtn").onclick = () => search();
  document.addEventListener("keydown", e => { if (e.key === "Enter" && !$("#goBtn").disabled && document.activeElement.tagName !== "INPUT") search(); });

  async function search() {
    if (!S.blob || !S.bbox) return;
    const btn = $("#goBtn");
    btn.disabled = true; $(".spinner", btn).classList.remove("hidden"); $(".lbl", btn).textContent = "Ищем…";
    setVerdict("busy");
    const f = new FormData();
    f.append("file", S.blob, "frame.jpg");
    for (const k of ["x", "y", "w", "h"]) f.append(k, Math.round(S.bbox[k]));
    f.append("top_k", topk.value);
    f.append("threshold", thr.value);
    f.append("exclude_same_camera", $("#xcam").checked);
    try {
      const r = await api("/api/search", { method: "POST", body: f });
      S.result = r; S.explain.clear();
      setVerdict(r);
      renderGrid(r);
      if (r.ranking.length) select(0); else $("#compare").classList.add("hidden");
      if (innerWidth < 1080) $("#verdict").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      setVerdict({ error: e.message });
      toast(e.message, "error");
    } finally {
      btn.disabled = false; $(".spinner", btn).classList.add("hidden"); $(".lbl", btn).textContent = "Найти машину";
      readyState();
    }
  }

  function setVerdict(r) {
    const v = $("#verdict"), ring = $("#ringVal");
    const set = (cls, title, sub, val, num) => {
      v.className = `card verdict ${cls}`;
      $("#vTitle").textContent = title;
      $("#vSub").textContent = sub;
      ring.style.strokeDashoffset = 326.7 * (1 - Math.max(0, Math.min(1, val)));
      $("#ringNum").textContent = num;
    };
    $("#route").classList.add("hidden");
    if (r === "busy") return set("busy", "Ищем…", "Строим отпечаток машины и сравниваем с базой.", 0, "…");
    if (r.error) return set("err", "Ошибка", r.error, 0, "!");
    if (!r.ranking.length) return set("no", "База пуста", "В галерее нет машин с других камер.", 0, "—");
    const top = r.ranking[0];
    $("#routeFrom").textContent = camText(r.query_camera);
    $("#routeTo").textContent = camText(top.camera);
    $("#route").classList.remove("hidden");
    if (r.refused) {
      set("no", "Уверенного совпадения нет", `Лучшая близость ${top.score.toFixed(3)} ниже порога ${r.threshold.toFixed(2)} — сервис отказывается угадывать. ${Math.round(r.ms)} мс.`,
        top.score, top.score.toFixed(2));
    } else {
      const c = r.candidates[0];
      set("ok", "Кандидат найден", `Близость ${c.score.toFixed(3)} (${bandName[band(c.score, r.threshold)]}) при пороге ${r.threshold.toFixed(2)}. ${Math.round(r.ms)} мс на сервере.`,
        c.score, c.score.toFixed(2));
    }
  }

  function renderGrid(r) {
    const g = $("#grid");
    g.textContent = "";
    $("#results").classList.toggle("hidden", !r.ranking.length);
    $("#resTitle").textContent = $("#xcam").checked ? "Похожие машины с других камер" : "Похожие машины (все камеры)";
    r.ranking.forEach((m, i) => {
      const b = el("button", `cand ${band(m.score, r.threshold)}`);
      b.type = "button";
      b.style.animationDelay = `${Math.min(i, 12) * 25}ms`;
      const th = el("div", "thumb");
      const im = document.createElement("img");
      im.loading = "lazy"; im.alt = `кандидат ${i + 1}`; im.src = m.crop_url;
      th.append(im, el("span", "rank", `#${i + 1}`));
      const meta = el("div", "meta");
      const row = el("div", "row");
      row.append(el("b", null, m.score.toFixed(3)), el("span", null, camText(m.camera)));
      const bar = el("div", "bar"), fill = el("i");
      fill.style.width = `${Math.max(0, Math.min(1, m.score)) * 100}%`;
      bar.appendChild(fill);
      meta.append(row, bar);
      b.append(th, meta);
      b.onclick = () => { select(i); $("#compare").scrollIntoView({ behavior: "smooth", block: "nearest" }); };
      g.appendChild(b);
    });
  }

  // ---------------------------------------------------------------- сравнение и карта внимания
  function select(i) {
    S.active = i;
    const r = S.result, m = r.ranking[i];
    document.querySelectorAll("#grid .cand").forEach((x, k) => x.classList.toggle("active", k === i));
    $("#compare").classList.remove("hidden");
    $("#cmpTitle").textContent = `Запрос и кандидат №${i + 1}`;
    $("#cmpScore").textContent = m.score.toFixed(3);
    const bd = band(m.score, r.threshold), pill = $("#cmpPill");
    pill.className = `pill ${bd}`; pill.textContent = bandName[bd];
    $("#camQ").textContent = camText(r.query_camera);
    $("#camC").textContent = camText(m.camera);
    $("#idC").textContent = m.id;
    showPlain();
    if ($("#heatToggle").checked) showExplain();
  }
  function showPlain() {
    const m = S.result && S.result.ranking[S.active];
    for (const [shot, img, src] of [["#shotQ", "#imgQ", S.queryCrop], ["#shotC", "#imgC", m && m.crop_url]]) {
      const s = $(shot);
      s.classList.remove("explained"); s.style.width = s.style.height = "";
      $(img).src = src || "";
      $(".heat", s).style.opacity = 0;
    }
  }
  function sizeShot(shot, aspect) {                 // рамка точно по пропорциям кропа
    const s = $(shot), box = s.parentElement.getBoundingClientRect();
    const maxW = box.width || 300, maxH = 260;
    let w = maxH * aspect, h = maxH;
    if (w > maxW) { w = maxW; h = maxW / aspect; }
    s.style.width = `${Math.round(w)}px`; s.style.height = `${Math.round(h)}px`; s.style.margin = "0 auto";
    s.classList.add("explained");
  }
  async function showExplain() {
    const m = S.result && S.result.ranking[S.active];
    if (!m) return;
    const key = `${S.token}|${JSON.stringify(S.bbox)}|${m.id}`;
    $("#heatBar").classList.remove("hidden");
    try {
      let d = S.explain.get(key);
      if (!d) {
        $("#heatNote").textContent = "считаем карту внимания…";
        const f = new FormData();
        f.append("file", S.blob, "frame.jpg");
        f.append("gallery_id", m.id);
        for (const k of ["x", "y", "w", "h"]) f.append(k, Math.round(S.bbox[k]));
        d = await api("/api/explain", { method: "POST", body: f });
        S.explain.set(key, d);
      }
      if (!$("#heatToggle").checked || S.result.ranking[S.active] !== m) return;
      const op = $("#heatOpacity").value;
      for (const [shot, img, part] of [["#shotQ", "#imgQ", d.query], ["#shotC", "#imgC", d.candidate]]) {
        sizeShot(shot, part.aspect);
        $(img).src = part.image;                    // квадрат модели растягивается обратно в пропорции кропа
        const h = $(".heat", $(shot));
        h.style.backgroundImage = `url(${part.heat})`;
        h.style.opacity = op;
      }
      $("#heatNote").textContent = HEAT_NOTE;
    } catch (e) {
      $("#heatToggle").checked = false;
      $("#heatBar").classList.add("hidden");
      toast(`Карта внимания: ${e.message}`, "error");
    }
  }
  const HEAT_NOTE = "Яркое — участки 14×14 px, которые сильнее всего совпали у двух снимков (фары, решётка, колёса, стойки). " +
    "Точное разложение близости ViT-модели по участкам; показаны 40% самых значимых — слабый ровный вклад фона и асфальта скрыт.";
  $("#heatToggle").onchange = e => {
    if (e.target.checked) showExplain();
    else { $("#heatBar").classList.add("hidden"); showPlain(); }
  };
  $("#heatOpacity").oninput = e => document.querySelectorAll(".shot .heat").forEach(h => { if (h.style.backgroundImage) h.style.opacity = e.target.value; });
  window.addEventListener("resize", () => { if ($("#heatToggle").checked && S.result) showExplain(); });

  // ---------------------------------------------------------------- экспорт
  $("#expJson").onclick = () => S.result && download("reid_result.json", JSON.stringify({ query_bbox: S.bbox, ...S.result }, null, 1), "application/json");
  $("#expCsv").onclick = () => {
    if (!S.result) return;
    const rows = ["rank,gallery_id,label,camera,score,accepted"].concat(
      S.result.ranking.map((m, i) => [i + 1, m.id, m.label || "", m.camera ?? "", m.score, m.accepted].join(",")));
    download("reid_result.csv", rows.join("\n"), "text/csv");
  };

  // ---------------------------------------------------------------- галерея
  async function loadGallery(reset) {
    if (reset) { S.galOffset = 0; $("#galGrid").textContent = ""; }
    const g = $("#galGrid");
    try {
      const items = await api(`/api/gallery?limit=${S.galLimit}&offset=${S.galOffset}`);
      items.forEach(it => {
        const c = el("div", "cand");
        const th = el("div", "thumb");
        const im = document.createElement("img");
        im.loading = "lazy"; im.alt = `ТС ${it.id}`; im.src = `/api/gallery/${it.id}/crop`;
        th.append(im, el("span", "rank", `#${it.id}`));
        const meta = el("div", "meta"), row = el("div", "row");
        row.append(el("span", null, camText(it.camera)), el("span", null, it.source === "test_query" ? "запрос" : it.source === "upload" ? "загружено" : "галерея"));
        meta.appendChild(row);
        c.append(th, meta);
        g.appendChild(c);
      });
      S.galOffset += items.length;
      $("#galMore").classList.toggle("hidden", items.length < S.galLimit);
    } catch (e) {
      g.textContent = `Галерея недоступна: ${e.message}`;
    }
  }
  $("#galMore").onclick = () => loadGallery(false);
  $("#galClear").onclick = async () => {
    if (!(await confirmBox("Очистить галерею?", "Все объекты базы будут удалены."))) return;
    try { await api("/api/gallery", { method: "DELETE" }); toast("Галерея очищена", "ok"); await health(); loadGallery(true); }
    catch (e) { toast(e.message, "error"); }
  };
  $("#galLoadDemo").onclick = async () => {
    try {
      await api("/api/demo/load", { method: "POST" });
      toast("Загружаем тестовую галерею…");
      const t = setInterval(async () => {
        const s = await api("/api/demo/status").catch(() => null);
        if (!s) return;
        $("#galInfo").textContent = s.error ? `Ошибка: ${s.error}` : `загружено ${s.done} из ${s.total}`;
        if (!s.running) { clearInterval(t); await health(); loadGallery(true); }
      }, 1500);
    } catch (e) { toast(e.message, "error"); }
  };

  // ---------------------------------------------------------------- старт
  thr.oninput(); topk.oninput();
  health();
  setInterval(health, 20000);
  loadSamples();
})();
