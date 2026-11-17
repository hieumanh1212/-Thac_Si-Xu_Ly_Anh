(() => {
  const $ = (id) => document.getElementById(id);
  const RING_LEN = 2 * Math.PI * 52;

  let currentImage = null; // File/Blob hoặc data URI
  let stream = null;
  let steps = [];
  let stepIdx = 0;
  let lastResult = null;
  const history = [];

  // ================= Toast =================
  function toast(msg, type = "info", ms = 4200) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    $("toasts").appendChild(el);
    setTimeout(() => { el.classList.add("hide"); setTimeout(() => el.remove(), 300); }, ms);
  }

  // ================= Trạng thái OCR =================
  async function pollStatus() {
    const badge = $("status");
    const label = badge.querySelector("span:last-child");
    try {
      const s = await (await fetch("/api/status")).json();
      if (s.ocr_ready) {
        badge.className = "pill pill-ok";
        label.textContent = "OCR sẵn sàng";
        return;
      }
      if (s.ocr_error) {
        badge.className = "pill pill-err";
        label.textContent = "Lỗi OCR";
        badge.title = s.ocr_error;
        return;
      }
      badge.className = "pill pill-wait";
      label.textContent = "Đang nạp mô hình OCR";
    } catch {
      badge.className = "pill pill-err";
      label.textContent = "Mất kết nối";
    }
    setTimeout(pollStatus, 2500);
  }
  pollStatus();

  // ================= Ảnh mẫu =================
  async function loadSamples() {
    try {
      const files = await (await fetch("/api/samples")).json();
      if (!files.length) return;
      const grid = $("samples");
      files.slice(0, 8).forEach((name) => {
        const b = document.createElement("button");
        b.type = "button";
        b.title = name;
        b.innerHTML = `<img alt="" loading="lazy">`;
        b.querySelector("img").src = `/samples/${encodeURIComponent(name)}`;
        b.addEventListener("click", async () => {
          const blob = await (await fetch(`/samples/${encodeURIComponent(name)}`)).blob();
          switchTab("upload");
          setImage(new File([blob], name, { type: blob.type }));
          recognize();
        });
        grid.appendChild(b);
      });
      $("samplesBox").hidden = false;
    } catch { /* không có ảnh mẫu cũng không sao */ }
  }
  loadSamples();

  // ================= Tabs =================
  function switchTab(name) {
    document.querySelectorAll(".seg").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $("panel-upload").hidden = name !== "upload";
    $("panel-camera").hidden = name !== "camera";
    if (name !== "camera") stopCamera();
  }
  document.querySelectorAll(".seg").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

  // ================= Chọn ảnh =================
  function setImage(fileOrUri) {
    currentImage = fileOrUri;
    $("preview").src = typeof fileOrUri === "string" ? fileOrUri : URL.createObjectURL(fileOrUri);
    $("previewWrap").hidden = false;
    $("dropHint").hidden = true;
    $("btnRecognize").disabled = false;
  }
  function clearImage() {
    currentImage = null;
    $("fileInput").value = "";
    $("previewWrap").hidden = true;
    $("dropHint").hidden = false;
    $("btnRecognize").disabled = true;
  }
  $("btnClear").addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); clearImage(); });
  $("fileInput").addEventListener("change", (e) => { if (e.target.files[0]) setImage(e.target.files[0]); });

  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) setImage(f);
    else toast("Vui lòng thả một file ảnh.", "err");
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) { switchTab("upload"); setImage(item.getAsFile()); toast("Đã dán ảnh từ clipboard."); }
  });

  // ================= Webcam =================
  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false,
      });
      $("video").srcObject = stream;
      $("camPlaceholder").hidden = true;
      $("btnCapture").disabled = false;
      $("btnCamStart").textContent = "Tắt camera";
    } catch (err) {
      toast("Không mở được camera: " + err.message, "err");
    }
  }
  function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    $("camPlaceholder").hidden = false;
    $("btnCapture").disabled = true;
    $("btnCamStart").textContent = "Bật camera";
  }
  $("btnCamStart").addEventListener("click", () => (stream ? stopCamera() : startCamera()));
  $("btnCapture").addEventListener("click", () => {
    const v = $("video"), c = $("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    setImage(c.toDataURL("image/jpeg", 0.92));
    recognize();
  });

  // ================= Nhận dạng =================
  function setBusy(busy) {
    $("btnRecognize").classList.toggle("busy", busy);
    $("btnRecognize").disabled = busy || !currentImage;
    $("previewWrap").classList.toggle("scanning", busy);
    $("loading").hidden = !busy;
    if (busy) { $("emptyState").hidden = true; $("result").hidden = true; }
  }

  async function recognize() {
    if (!currentImage) return;
    setBusy(true);
    try {
      let resp;
      if (typeof currentImage === "string") {
        resp = await fetch("/api/recognize", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: currentImage }),
        });
      } else {
        const fd = new FormData();
        fd.append("image", currentImage);
        resp = await fetch("/api/recognize", { method: "POST", body: fd });
      }
      const data = await resp.json();
      if (!resp.ok || data.error) throw new Error(data.error || resp.statusText);
      render(data);
      addHistory(data);
      if (data.ocr_error) toast(data.ocr_error, "err", 7000);
      else if (!data.found) toast("Không tìm thấy biển số trong ảnh.", "err");
    } catch (err) {
      toast(err.message, "err");
      if (!lastResult) $("emptyState").hidden = false;
      else $("result").hidden = false;
    } finally {
      setBusy(false);
    }
  }
  $("btnRecognize").addEventListener("click", recognize);

  // ================= Hiển thị kết quả =================
  function render(d) {
    lastResult = d;
    $("emptyState").hidden = true;
    $("result").hidden = false;
    // Kích hoạt lại animation
    const r = $("result"); r.style.animation = "none"; void r.offsetWidth; r.style.animation = "";
    const p = $("plate"); p.style.animation = "none"; void p.offsetWidth; p.style.animation = "";

    $("plateLine1").textContent = d.found ? d.line1 || "?" : "--";
    $("plateLine2").textContent = d.found ? d.line2 || "?" : "--";
    p.classList.toggle("invalid", !d.valid);

    const vb = $("validBadge");
    vb.className = "pill " + (d.valid ? "valid" : "invalid");
    vb.innerHTML = `<i class="dot"></i>${d.valid ? "Đúng định dạng biển số VN" : d.found ? "Chưa khớp định dạng" : "Không tìm thấy biển số"}`;

    const conf = d.found ? d.confidence * 100 : 0;
    animateNumber($("confVal"), conf);
    const ring = $("ringFg");
    ring.style.strokeDashoffset = RING_LEN * (1 - conf / 100);
    ring.style.stroke = conf >= 80 ? "var(--ok)" : conf >= 50 ? "var(--warn)" : "var(--err)";

    $("sTime").textContent = `${Math.round(d.time_ms.total)} ms`;
    $("sChars").textContent = d.num_chars;
    $("sCands").textContent = d.num_candidates;
    $("sRaw").textContent = d.raw.filter(Boolean).join(" · ") || "–";

    const total = d.time_ms.detect_segment + d.time_ms.ocr || 1;
    $("tDetect").style.width = (d.time_ms.detect_segment / total) * 100 + "%";
    $("tOcr").style.width = (d.time_ms.ocr / total) * 100 + "%";
    $("tDetectMs").textContent = `${Math.round(d.time_ms.detect_segment)} ms`;
    $("tOcrMs").textContent = `${Math.round(d.time_ms.ocr)} ms`;

    $("imgDetected").src = d.detected_image;
    $("imgPlate").src = d.plate_image;
    const chars = $("imgChars").closest("figure");
    chars.hidden = !d.chars_image;
    if (d.chars_image) $("imgChars").src = d.chars_image;

    $("btnCopy").hidden = !(d.found && d.text);
    renderSteps(d.steps);
  }

  function animateNumber(el, to) {
    const from = parseFloat(el.textContent) || 0;
    const t0 = performance.now();
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / 900);
      const e = 1 - Math.pow(1 - k, 3);
      el.textContent = (from + (to - from) * e).toFixed(1);
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  $("btnCopy").addEventListener("click", async () => {
    if (!lastResult) return;
    try {
      await navigator.clipboard.writeText(lastResult.text);
      toast(`Đã sao chép "${lastResult.text}"`, "ok");
    } catch { toast("Không sao chép được.", "err"); }
  });

  // ================= Pipeline viewer =================
  function renderSteps(list) {
    steps = list || [];
    $("pipeline").hidden = !steps.length;
    const thumbs = $("thumbs");
    thumbs.innerHTML = "";
    steps.forEach((s, i) => {
      const b = document.createElement("button");
      b.className = "thumb";
      b.type = "button";
      b.innerHTML = `<img alt=""><span><i></i></span>`;
      b.querySelector("img").src = s.image;
      b.querySelector("i").textContent = String(i + 1).padStart(2, "0");
      b.querySelector("span").append(s.title);
      b.addEventListener("click", () => showStep(i));
      thumbs.appendChild(b);
    });
    // Mặc định mở bước "Phân đoạn ký tự" nếu có, cho người xem thấy ngay phần thú vị
    const seg = steps.findIndex((s) => s.title.startsWith("Phân đoạn"));
    showStep(seg >= 0 ? seg : 0, false);
  }

  function showStep(i, scroll = true) {
    if (!steps.length) return;
    stepIdx = (i + steps.length) % steps.length;
    const s = steps[stepIdx];
    const img = $("stepImg");
    img.src = s.image;
    img.dataset.caption = `${stepIdx + 1}. ${s.title}: ${s.desc}`;
    img.style.animation = "none"; void img.offsetWidth; img.style.animation = "";
    $("stepCount").textContent = `BƯỚC ${String(stepIdx + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
    $("stepTitle").textContent = s.title;
    $("stepDesc").textContent = s.desc;
    $("stepProgress").style.width = ((stepIdx + 1) / steps.length) * 100 + "%";
    const thumbs = [...$("thumbs").children];
    thumbs.forEach((t, k) => t.classList.toggle("active", k === stepIdx));
    if (scroll && thumbs[stepIdx]) thumbs[stepIdx].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }
  $("stepPrev").addEventListener("click", () => showStep(stepIdx - 1));
  $("stepNext").addEventListener("click", () => showStep(stepIdx + 1));

  // ================= Lịch sử =================
  function addHistory(d) {
    history.unshift(d);
    if (history.length > 12) history.pop();
    renderHistory();
  }
  function renderHistory() {
    $("historyBox").hidden = !history.length;
    const box = $("historyList");
    box.innerHTML = "";
    history.forEach((h) => {
      const b = document.createElement("button");
      b.className = "h-item";
      b.type = "button";
      b.innerHTML = `<img alt=""><div><div class="h-text"></div><div class="h-meta"><i class="d"></i><span></span></div></div>`;
      b.querySelector("img").src = h.plate_image;
      b.querySelector(".h-text").textContent = h.found ? h.text || "?" : "Không tìm thấy";
      b.querySelector(".d").style.background = h.valid ? "var(--ok)" : "var(--warn)";
      b.querySelector(".h-meta span").textContent = `${Math.round(h.time_ms.total)} ms · ${(h.confidence * 100).toFixed(0)}%`;
      b.addEventListener("click", () => { render(h); window.scrollTo({ top: 0, behavior: "smooth" }); });
      box.appendChild(b);
    });
  }
  $("btnClearHistory").addEventListener("click", () => { history.length = 0; renderHistory(); });

  // ================= Lightbox & phím tắt =================
  document.addEventListener("click", (e) => {
    const img = e.target.closest("img[data-zoom]");
    if (img && img.src) {
      $("lightboxImg").src = img.src;
      $("lightboxCaption").textContent = img.dataset.caption || img.alt || "";
      $("lightbox").hidden = false;
    }
  });
  $("lightbox").addEventListener("click", () => ($("lightbox").hidden = true));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("lightbox").hidden = true;
    if (e.target.matches("input, textarea, button")) return;
    if (!$("pipeline").hidden && e.key === "ArrowRight") showStep(stepIdx + 1);
    if (!$("pipeline").hidden && e.key === "ArrowLeft") showStep(stepIdx - 1);
    if (e.key === "Enter" && currentImage && !$("btnRecognize").disabled) recognize();
  });
})();
