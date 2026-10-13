const $ = (sel) => document.querySelector(sel);

async function getJSON(url) {
  const r = await fetch(url);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---------------------------------------------------------------- rtsp
async function refreshRtspStatus() {
  const data = await getJSON("/api/rtsp");
  if (document.activeElement !== $("#rtsp-input")) {
    $("#rtsp-input").value = data.source;
  }
  const badge = $("#conn-badge");
  if (data.connected) {
    badge.textContent = "Da ket noi camera";
    badge.className = "badge ok";
  } else {
    badge.textContent = "Mat ket noi camera";
    badge.className = "badge bad";
  }
}

$("#rtsp-save").addEventListener("click", async () => {
  const source = $("#rtsp-input").value.trim();
  if (!source) return;
  await postJSON("/api/rtsp", { source });
  setTimeout(refreshRtspStatus, 500);
});

// ---------------------------------------------------------------- enroll
$("#enroll-start").addEventListener("click", async () => {
  const name = $("#enroll-name").value.trim();
  const target = parseInt($("#enroll-target").value, 10) || 15;
  if (!name) {
    alert("Vui long nhap ten nguoi can dang ky");
    return;
  }
  const auto = $("#enroll-auto").checked;
  const res = await postJSON("/api/enroll/start", { name, target, auto });
  if (res.error) {
    alert(res.error);
    return;
  }
  $("#enroll-capture-msg").textContent = "";
  $("#enroll-progress-wrap").classList.remove("hidden");
});

$("#enroll-cancel").addEventListener("click", async () => {
  await postJSON("/api/enroll/cancel");
});

$("#enroll-auto").addEventListener("change", async () => {
  await postJSON("/api/enroll/auto", { auto: $("#enroll-auto").checked });
});

$("#enroll-capture").addEventListener("click", async () => {
  const btn = $("#enroll-capture");
  const msg = $("#enroll-capture-msg");
  btn.disabled = true;
  msg.className = "capture-msg";
  msg.textContent = "Dang chup...";
  try {
    const res = await postJSON("/api/enroll/capture");
    msg.textContent = res.msg;
    msg.classList.add(res.ok ? "ok-text" : "bad-text");
  } catch (e) {
    msg.textContent = "Loi ket noi server";
    msg.classList.add("bad-text");
  }
  btn.disabled = false;
  pollEnrollStatus();
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderLiveStatus(s) {
  const live = $("#enroll-live");
  const c = s.last_check;
  if (!s.active) {
    live.textContent = "";
    live.className = "live-status";
    return;
  }
  if (!s.auto && !c) {
    live.textContent = "Tu dong chup dang TAT - bam 'Chup ngay' khi co mat";
    live.className = "live-status";
    return;
  }
  if (!c || c.age > 3) {
    live.textContent = "Dang cho frame tu camera / AI...";
    live.className = "live-status";
    return;
  }
  const yaw = c.yaw != null ? `, yaw ${c.yaw > 0 ? "+" : ""}${c.yaw}, pitch ${c.pitch > 0 ? "+" : ""}${c.pitch}` : "";
  const metrics = c.size != null
    ? ` [mat ${c.size}px, tin cay ${c.score}${yaw}, do net ${c.sharpness ?? "-"}, ${c.faces} mat]`
    : "";
  if (c.ok) {
    live.textContent = "Dat dieu kien" + metrics;
    live.className = "live-status ok";
  } else {
    live.textContent = "Chua chup: " + c.reason + metrics;
    live.className = "live-status bad";
  }
}

function renderPoses(s) {
  const hint = $("#enroll-hint");
  if (s.active && s.hint) {
    hint.textContent = "Huong dan: " + s.hint;
    hint.classList.remove("hidden");
  } else {
    hint.classList.add("hidden");
  }
  $("#enroll-poses").innerHTML = (s.poses || [])
    .filter((p) => p.quota > 0)
    .map((p) => {
      const done = p.captured >= p.quota;
      const cls = done ? "done" : p.key === s.next_pose ? "current" : "";
      return `<span class="pose-chip ${cls}">${done ? "&#10003; " : ""}${escapeHtml(p.label)} ${p.captured}/${p.quota}</span>`;
    })
    .join("");
}

function renderCounts(counts) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  $("#enroll-counts").innerHTML = entries.length
    ? "Bi loai: " + entries.map(([k, v]) => `${escapeHtml(k)} <b>${v}</b>`).join(" · ")
    : "";
}

function renderLog(log) {
  $("#enroll-log").innerHTML = (log || [])
    .map((e) => `<div class="log-line ${e.level}"><span>${e.t}</span> ${escapeHtml(e.msg)}</div>`)
    .join("");
}

let wasEnrollActive = false;
let lastThumb = null;
async function pollEnrollStatus() {
  const s = await getJSON("/api/enroll/status");
  $("#enroll-capture").disabled = !s.active;
  if (document.activeElement !== $("#enroll-auto") && s.active) {
    $("#enroll-auto").checked = s.auto;
  }
  if (s.active || s.log.length) {
    $("#enroll-progress-wrap").classList.remove("hidden");
    const pct = s.target ? Math.round((s.captured / s.target) * 100) : 0;
    $("#enroll-progress-bar").style.width = pct + "%";
    $("#enroll-progress-text").textContent = `${s.name || ""}: da quet ${s.captured}/${s.target} anh`;
    if (s.thumb && s.thumb !== lastThumb) {
      lastThumb = s.thumb;
      $("#enroll-thumb").src = s.thumb + "?t=" + Date.now();
      $("#enroll-thumb").classList.remove("hidden");
    }
    renderPoses(s);
    renderLiveStatus(s);
    renderCounts(s.reject_counts);
    renderLog(s.log);
    if (!s.active && wasEnrollActive) {
      loadPeople();
    }
  }
  wasEnrollActive = s.active;
}

// ---------------------------------------------------------------- people
async function loadPeople() {
  const data = await getJSON("/api/identities");
  const tbody = $("#people-table tbody");
  tbody.innerHTML = "";
  data.people.forEach((p) => {
    const tr = document.createElement("tr");
    const local = p.source === "local";
    const action = local
      ? `<button class="del-btn" data-name="${escapeHtml(p.name)}">Xoa</button>`
      : `<span class="src-tag" title="${escapeHtml(p.source)}">import</span>`;
    tr.innerHTML = `<td>${escapeHtml(p.name)}</td><td>${p.num_images}</td>
      <td>${escapeHtml(p.created_at || "")}</td><td>${action}</td>`;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll(".del-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!confirm(`Xoa danh tinh "${btn.dataset.name}"?`)) return;
      await fetch("/api/identities/" + encodeURIComponent(btn.dataset.name), { method: "DELETE" });
      loadPeople();
    });
  });
}

// ---------------------------------------------------------------- settings
const SETTING_KEYS = [
  "person_conf_threshold",
  "face_conf_threshold",
  "min_face_size",
  "similarity_threshold",
  "process_every_n_frames",
  "recheck_interval_frames",
  "enroll_min_face_size",
  "enroll_min_sharpness",
  "gallery_mode",
];
// Select (chuoi) thay vi thanh truot (so).
const isSelect = (input) => input.tagName === "SELECT";

async function loadSettings() {
  const data = await getJSON("/api/settings");
  SETTING_KEYS.forEach((key) => {
    const input = $("#s-" + key);
    input.value = data[key];
    if (!isSelect(input)) input.nextElementSibling.textContent = data[key];
  });
}

SETTING_KEYS.forEach((key) => {
  const input = $("#s-" + key);
  input.addEventListener("input", () => {
    if (!isSelect(input)) input.nextElementSibling.textContent = input.value;
  });
});

$("#settings-save").addEventListener("click", async () => {
  const payload = {};
  SETTING_KEYS.forEach((key) => {
    const input = $("#s-" + key);
    payload[key] = isSelect(input) ? input.value : parseFloat(input.value);
  });
  await postJSON("/api/settings", payload);
});

// ---------------------------------------------------------------- init
refreshRtspStatus();
loadPeople();
loadSettings();
setInterval(refreshRtspStatus, 3000);
setInterval(pollEnrollStatus, 600);
