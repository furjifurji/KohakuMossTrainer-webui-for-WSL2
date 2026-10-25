"use strict";
const $ = (s, r = document) => r.querySelector(s);
const state = { schema: null, params: {}, presets: [], current: "", logNext: 0, connected: false, running: false, derived: {} };

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "tip") e.dataset.tip = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
  return e;
}
function fields() { return state.schema.tabs.flatMap(t => t.sections.flatMap(s => s.fields)); }
function defaults() { return Object.fromEntries(fields().map(f => [f.key, JSON.parse(JSON.stringify(f.default))])); }

function showToast(msg, type = "ok") {
  let c = $("#toast-container");
  if (!c) {
    c = el("div", { id: "toast-container", class: "toast-container" });
    document.body.append(c);
  }
  const t = el("div", { class: "toast" + (type !== "ok" ? " toast-" + type : "") }, msg);
  c.append(t);
  setTimeout(() => t.classList.add("show"), 10);
  setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 300);
  }, 4000);
}

// ------------------------------------------------------------------ tooltip
const tipBox = $("#tooltip");
document.addEventListener("mouseover", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) return;
  tipBox.textContent = t.dataset.tip;
  tipBox.classList.remove("hidden");
});
document.addEventListener("mousemove", e => {
  if (tipBox.classList.contains("hidden")) return;
  const w = tipBox.offsetWidth, h = tipBox.offsetHeight;
  let x = e.clientX + 16, y = e.clientY + 18;
  if (x + w > innerWidth - 8) x = e.clientX - w - 12;
  if (y + h > innerHeight - 8) y = e.clientY - h - 12;
  tipBox.style.left = Math.max(4, x) + "px";
  tipBox.style.top = Math.max(4, y) + "px";
});
document.addEventListener("mouseout", e => { if (e.target.closest("[data-tip]")) tipBox.classList.add("hidden"); });

// -------------------------------------------------------------------- form
function labelOf(f) {
  return el("label", { tip: `${f.label}（${f.zh}）\n${f.tip}` }, f.label, el("span", { class: "zh" }, f.zh));
}
function bind(input, f, read) {
  input.addEventListener("input", () => { state.params[f.key] = read(); onChange(f.key); });
  input.addEventListener("change", () => { state.params[f.key] = read(); onChange(f.key); });
}
function optTip(f, v) {
  const o = (f.options || []).find(o => String(o.value) === String(v));
  return o ? `${o.label}：${o.tip}` : "";
}
function buildField(f) {
  const wrap = el("div", { class: "field" + (f.wide ? " wide" : "") + (f.type === "bool" ? " bool" : ""), "data-key": f.key });
  const val = state.params[f.key];
  if (f.type === "bool") {
    const cb = el("input", { type: "checkbox" });
    cb.checked = !!val;
    bind(cb, f, () => cb.checked);
    wrap.append(el("label", { tip: `${f.label}（${f.zh}）\n${f.tip}` }, cb, f.label, el("span", { class: "zh" }, f.zh)));
    return wrap;
  }
  wrap.append(labelOf(f));
  if (f.type === "select") {
    const s = el("select", {}, f.options.map(o => el("option", { value: o.value, tip: o.tip, title: o.tip }, o.label)));
    s.value = val;
    const hint = el("div", { class: "opt-tip" }, optTip(f, val));
    bind(s, f, () => { hint.textContent = optTip(f, s.value); return s.value; });
    wrap.append(s, hint);
  } else if (f.type === "number") {
    const i = el("input", { type: "number", min: f.min, max: f.max, step: f.step ?? "any" });
    i.value = val;
    bind(i, f, () => (i.value === "" ? "" : Number(i.value)));
    wrap.append(i);
  } else if (f.type === "textarea") {
    const t = el("textarea", { placeholder: f.placeholder || "" });
    t.value = val;
    bind(t, f, () => t.value);
    wrap.append(t);
  } else if (f.type === "datasets") {
    wrap.append(buildDatasets(f));
  } else {
    const i = el("input", { type: "text", placeholder: f.placeholder || "" });
    i.value = val ?? "";
    bind(i, f, () => i.value);
    if (f.type === "path" && f.scan) {
      const pick = el("select", { tip: "从权重根目录下已扫描到的文件中选择" }, el("option", { value: "" }, "📂 扫描选择…"));
      pick.addEventListener("focus", () => scanInto(pick, f));
      pick.addEventListener("change", () => { if (pick.value) { i.value = pick.value; state.params[f.key] = i.value; onChange(f.key); } });
      wrap.append(el("div", { class: "path-row" }, i, pick));
    } else wrap.append(i);
  }
  return wrap;
}
async function scanInto(sel, f) {
  const root = state.params.weights_root;
  try {
    const r = await api(`/api/scan?root=${encodeURIComponent(root)}&sub=${encodeURIComponent(f.scan)}`);
    sel.replaceChildren(el("option", { value: "" }, r.files.length ? "📂 选择文件…" : "（未找到 safetensors）"),
      ...r.files.map(p => el("option", { value: p }, p.split(/[\\/]/).slice(-2).join("/"))));
  } catch (e) { /* ignore */ }
}
function buildDatasets(f) {
  const box = el("div");
  const render = () => {
    box.replaceChildren(
      el("div", { class: "ds-head" }, el("span", { tip: "图片所在文件夹（可粘贴 Windows 路径）" }, "Path 路径"),
        el("span", { tip: "kohya：子文件夹形如 10_名称（10 为重复次数）\nfolder：目录内直接是图片+同名 txt" }, "Structure 结构"),
        el("span", { tip: "Weight：该数据集的整体重复倍率（整数防呆，1 = 不变，2 = 2倍采样）" }, "Weight 倍率"), el("span")));
    state.params.datasets.forEach((d, idx) => {
      const p = el("input", { type: "text", placeholder: "F:\\path\\to\\dataset" });
      p.value = d.path || "";
      p.addEventListener("input", () => { d.path = p.value; onChange("datasets"); });
      const s = el("select", {}, el("option", { value: "kohya" }, "kohya (N_名称)"), el("option", { value: "folder" }, "folder (图+txt)"));
      s.value = d.source || "kohya";
      s.addEventListener("change", () => { d.source = s.value; onChange("datasets"); });
      const w = el("input", { type: "number", step: "1", min: "1" });
      w.value = Math.max(1, Math.round(Number(d.weight) || 1));
      const syncWeight = () => {
        const val = Math.max(1, Math.round(Number(w.value) || 1));
        w.value = val;
        d.weight = val;
        onChange("datasets");
      };
      w.addEventListener("input", syncWeight);
      w.addEventListener("blur", syncWeight);
      const x = el("button", { class: "btn btn-danger btn-sm", tip: "删除该数据集", onclick: () => { state.params.datasets.splice(idx, 1); render(); onChange("datasets"); } }, "✕");
      box.append(el("div", { class: "ds-item" }, p, s, w, x));
    });
    box.append(el("button", { class: "btn btn-secondary btn-sm", tip: "添加一个数据集目录", onclick: () => { state.params.datasets.push({ path: "", source: "kohya", weight: 1 }); render(); } }, "+ Add Dataset 添加"));
  };
  render();
  return box;
}
function visible(f) {
  if (!f.show) return true;
  return Object.entries(f.show).every(([k, v]) => (Array.isArray(v) ? v : [v]).includes(state.params[k]));
}
function applyVisibility() {
  for (const f of fields()) {
    const e = $(`.field[data-key="${f.key}"]`);
    if (e) e.classList.toggle("hidden", !visible(f));
  }
  for (const s of document.querySelectorAll(".section")) {
    s.classList.toggle("hidden", !s.querySelector(".field:not(.hidden)"));
  }
}

function buildUI() {
  const tabs = $("#tabs"), panes = $("#panes");
  tabs.replaceChildren(); panes.replaceChildren();
  state.schema.tabs.forEach((t, i) => {
    tabs.append(el("button", { class: "tab" + (i ? "" : " active"), "data-tab": t.id, tip: `${t.label}（${t.zh}）\n${t.tip}`, onclick: () => showTab(t.id) }, t.label, el("span", { class: "zh" }, t.zh)));
    const pane = el("div", { class: "pane" + (i ? "" : " active"), id: "pane-" + t.id });
    for (const s of t.sections) {
      pane.append(el("div", { class: "section" },
        el("h4", { tip: `${s.title}（${s.zh}）\n${s.tip}` }, s.title, el("span", { class: "zh" }, s.zh)),
        el("div", { class: "grid" }, s.fields.map(buildField))));
    }
    panes.append(pane);
  });
  tabs.append(el("button", { class: "tab", "data-tab": "console", tip: "Console（控制台）\n实时日志、数据集扫描结果与训练状态。", onclick: () => showTab("console") }, "Console", el("span", { class: "zh" }, "控制台")));
  panes.append(el("div", { class: "pane", id: "pane-console" },
    el("div", { class: "console-wrap" },
      el("div", { id: "stats", class: "info-grid" }),
      el("div", { id: "train-dash", class: "train-dash hidden" },
        el("div", { class: "dash-top" },
          el("div", { class: "dash-title-group" },
            el("span", { id: "dash-dot", class: "pulse-dot" }),
            el("span", { id: "dash-status-text", class: "dash-status-text" }, "训练进行中")
          ),
          el("div", { class: "dash-badges" },
            el("span", { id: "dash-epoch", class: "dash-badge badge-epoch" }, "Epoch 0 / 0"),
            el("span", { id: "dash-step", class: "dash-badge badge-step" }, "Step 0 / 0 (0.0%)")
          )
        ),
        el("div", { class: "progress-track" },
          el("div", { id: "progress-fill", class: "progress-fill" })
        ),
        el("div", { class: "dash-metrics" },
          el("div", { class: "metric-item" },
            el("span", { class: "metric-lbl" }, "Loss 损失"),
            el("span", { id: "dash-loss", class: "metric-val" }, "-")
          ),
          el("div", { class: "metric-item" },
            el("span", { class: "metric-lbl" }, "Learning Rate 学习率"),
            el("span", { id: "dash-lr", class: "metric-val" }, "-")
          ),
          el("div", { class: "metric-item" },
            el("span", { class: "metric-lbl" }, "Speed 速度"),
            el("span", { id: "dash-speed", class: "metric-val" }, "-")
          ),
          el("div", { class: "metric-item" },
            el("span", { class: "metric-lbl" }, "Elapsed / ETA 耗时与预估"),
            el("span", { id: "dash-time", class: "metric-val" }, "-")
          )
        )
      ),
      el("div", { class: "console-bar" }, el("span", { id: "stage", class: "badge" }, "空闲"),
        el("span", {}, el("button", { class: "btn btn-secondary btn-sm", tip: "清空日志", onclick: clearLogs }, "清空 Clear"))),
      el("div", { id: "console" }))));
  applyVisibility();
}
function showTab(id) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === id));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("active", p.id === "pane-" + id));
  if (id === "console") { const c = $("#console"); c.scrollTop = c.scrollHeight; }
}

// ------------------------------------------------------------- state/params
let saveTimer = null, previewTimer = null;
function onChange(key) {
  applyVisibility();
  $("#title").textContent = state.params.run_name || "Untitled";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => localStorage.setItem("moss_params", JSON.stringify(state.params)), 300);
  clearTimeout(previewTimer);
  previewTimer = setTimeout(updateSummary, 400);
}
async function updateSummary() {
  try {
    const r = await api("/api/preview", { ...state.params });
    const d = r.derived;
    $("#summary").textContent = `总步数 ${d.steps} · 每 ${d.export_every} 步导出 LoRA · 每 ${d.save_every} 步保存检查点 · 有效批量 ${d.effective_batch} (BS ${d.batch_size} × 累加 ${d.grad_acc}) · 模式 ${d.resolution_mode} · 目标面积 ${d.target_area}px² · 宽高比 ${d.aspect_range[0]}~${d.aspect_range[1]}`;
  } catch (e) { $("#summary").textContent = ""; }
}
function setParams(p) {
  state.params = { ...defaults(), ...p };
  buildUI();
  onChange();
}
async function loadPresets() {
  try {
    state.presets = (await api("/api/presets")).presets || [];
  } catch (e) {
    state.presets = [];
  }
  const box = $("#preset-list");
  if (!state.presets.length) {
    box.replaceChildren(el("div", { class: "preset-empty" },
      el("div", { class: "empty-icon" }, "📁"),
      el("div", { class: "empty-title" }, "暂无保存的预设配置"),
      el("div", { class: "empty-tip" }, "点击上方 [+ New] 新建，或点击 [Save] 保存当前配置")
    ));
    return;
  }
  box.replaceChildren(...state.presets.map(n => el("div", {
    class: "preset" + (n === state.current ? " active" : ""),
    onclick: async () => {
      state.current = n;
      try {
        const p = await api("/api/preset?name=" + encodeURIComponent(n));
        setParams(p);
        loadPresets();
        showToast(`📂 已切换载入预设：【${n}】`);
      } catch (err) {
        showToast(`载入预设失败: ${err.message}`, "err");
      }
    }
  },
    el("span", { class: "preset-name", tip: `点击切换预设【${n}】\n文件对应: webui_app/presets/${n}.json` }, "📄 " + n),
    el("button", {
      class: "del",
      tip: `删除预设 ${n}`,
      onclick: async ev => {
        ev.stopPropagation();
        if (confirm(`确定要彻底删除预设【${n}】吗？\n（将删除 webui_app/presets/${n}.json 文件）`)) {
          await api("/api/preset_delete", { name: n });
          if (state.current === n) state.current = "";
          await loadPresets();
          showToast(`🗑️ 已删除预设：${n}`);
        }
      }
    }, "✕")
  )));
}

// ------------------------------------------------------------------ actions
async function action(path, body) {
  try { await api(path, body); showTab("console"); } catch (e) { alert(e.message); }
}
function stats(items) {
  $("#stats").replaceChildren(...items.map(([k, v]) => el("div", { class: "stat" }, el("b", {}, v), el("span", {}, k))));
}
async function showProbe() {
  const r = await api("/api/last_probe");
  if (!r || !r.catalog) return;
  const c = r.catalog, p = r.plan;
  stats([["图片数", c.rows], ["每 epoch 样本", Math.round(c.epoch_items)], ["每 epoch 批次", r.batches], ["每 epoch 步数", r.steps_per_epoch],
    ["批内样本均值", p.samples_per_batch.toFixed(2)], ["放大占比", (p.upscaled_share * 100).toFixed(1) + "%"], ["不同目标尺寸", r.distinct_targets],
    ["超长丢弃", p.overlong_dropped], ["未成批丢弃", p.unbatched]]);
}
function showConfig(text) {
  const m = el("div", { class: "modal", onclick: e => { if (e.target === m) m.remove(); } },
    el("div", { class: "modal-box" }, el("header", {}, el("b", {}, "生成的 KohakuMoss 配置"), el("button", { class: "btn btn-secondary btn-sm", onclick: () => m.remove() }, "关闭")), el("pre", {}, text)));
  document.body.append(m);
}
$("#btn-train").onclick = () => action("/api/train", { ...state.params });
$("#btn-probe").onclick = async () => { await action("/api/probe", { ...state.params }); };
$("#btn-doctor").onclick = () => action("/api/doctor", {});
$("#btn-stop").onclick = () => { if (confirm("确定终止当前任务？")) api("/api/stop", {}); };
$("#btn-preview").onclick = async () => { const r = await api("/api/preview", { ...state.params }); showConfig(r.config); };

$("#btn-new").onclick = async () => {
  const defaultName = "moss_config_" + (state.presets.length + 1);
  const name = prompt("【新建配置】请输入新配置名称（将在左侧列表中新增此配置）：", defaultName);
  if (!name || !name.trim()) return;
  const cleanName = name.trim();
  const newParams = { ...defaults(), run_name: cleanName };
  setParams(newParams);
  try {
    const r = await api("/api/preset", { name: cleanName, params: newParams });
    state.current = r.name;
    await loadPresets();
    showToast(`🎉 新配置【${r.name}】已成功创建！\n📁 已加入左侧列表并保存至: webui_app/presets/${r.name}.json`);
  } catch (err) {
    showToast(`创建失败: ${err.message}`, "err");
  }
};

$("#btn-save").onclick = async () => {
  let targetName = state.current;
  if (!targetName) {
    targetName = prompt("【保存配置】请输入预设名称（将保存并在左侧列表中显示）：", state.params.run_name || "moss_lora_preset");
    if (!targetName || !targetName.trim()) return;
    targetName = targetName.trim();
  }
  try {
    const r = await api("/api/preset", { name: targetName, params: state.params });
    state.current = r.name;
    await loadPresets();
    showToast(`💾 预设【${r.name}】保存成功！\n📁 文件已同步至: webui_app/presets/${r.name}.json`);
  } catch (err) {
    showToast(`保存失败: ${err.message}`, "err");
  }
};

// ----------------------------------------------------------------- polling
function colorLine(l) {
  if (l.startsWith(">>>")) return el("div", { class: "log-sys" }, l);
  if (/error|Traceback|Exception/i.test(l)) return el("div", { class: "log-err" }, l);
  return el("div", {}, l);
}
async function clearLogs() { await api("/api/clear_logs", {}); state.logNext = 0; $("#console").replaceChildren(); }
function setConn(ok, running, text) {
  const c = $("#conn");
  c.className = "conn " + (ok ? (running ? "busy" : "ok") : "bad");
  $("#conn-text").textContent = text;
}
function formatSec(s) {
  if (!s || s <= 0) return "00:00";
  const m = Math.floor(s / 60) % 60;
  const h = Math.floor(s / 3600);
  const sec = s % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}
let failures = 0;
async function poll() {
  try {
    const st = await api("/api/status");
    failures = 0;
    state.running = st.running;
    $("#btn-train").disabled = st.running;
    $("#btn-doctor").disabled = st.running;
    $("#btn-probe").disabled = st.running;
    $("#btn-stop").classList.toggle("hidden", !st.running);
    setConn(true, st.running, st.running ? `运行中：${st.task} ${st.stage ? "· " + st.stage : ""} (${st.elapsed}s)` : "WSL 服务已连接 · 空闲");
    const stage = $("#stage"); if (stage) stage.textContent = st.running ? `${st.task} · ${st.stage || ""}` : "空闲";

    if (st.progress && st.progress.total_steps > 0) {
      const p = st.progress;
      const dash = $("#train-dash");
      if (dash) {
        if (st.running || p.step > 0) dash.classList.remove("hidden");
        const dot = $("#dash-dot");
        if (dot) dot.className = "pulse-dot" + (st.running ? " active" : " done");
        const statusText = $("#dash-status-text");
        if (statusText) statusText.textContent = st.running ? (st.stage ? `训练进行中 · ${st.stage}` : "训练进行中…") : (p.step >= p.total_steps ? "训练已完成" : "任务已就绪");

        const dEpoch = $("#dash-epoch");
        if (dEpoch) dEpoch.textContent = `Epoch ${p.epoch} / ${p.total_epochs}`;

        const dStep = $("#dash-step");
        if (dStep) dStep.textContent = `Step ${p.step} / ${p.total_steps} (${p.percent}%)`;

        const fill = $("#progress-fill");
        if (fill) fill.style.width = p.percent + "%";

        if (p.loss && $("#dash-loss")) $("#dash-loss").textContent = p.loss;
        if (p.lr && $("#dash-lr")) $("#dash-lr").textContent = p.lr;
        if (p.speed && $("#dash-speed")) $("#dash-speed").textContent = p.speed;

        const timeBox = $("#dash-time");
        if (timeBox) {
          const elp = st.elapsed ? formatSec(st.elapsed) : "-";
          const eta = p.eta || "-";
          timeBox.textContent = `已用 ${elp} · 剩余约 ${eta}`;
        }
      }
    }

    const lg = await api("/api/logs?since=" + state.logNext);
    const out = $("#console");
    if (out) {
      if (lg.reset) out.replaceChildren();
      if (lg.lines.length) {
        const near = out.scrollHeight - out.scrollTop - out.clientHeight < 80;
        lg.lines.forEach(l => out.append(colorLine(l.replace(/\n$/, ""))));
        while (out.childNodes.length > 3000) out.removeChild(out.firstChild);
        if (near) out.scrollTop = out.scrollHeight;
      }
      state.logNext = lg.next;
    }
    if (!st.running && state.wasRunning) showProbe();
    state.wasRunning = st.running;
  } catch (e) {
    failures++;
    setConn(false, false, "无法连接 WSL 服务（重试中…）");
  }
  setTimeout(poll, state.running ? 1000 : 2500);
}

// -------------------------------------------------------------------- theme
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("moss_theme", theme);
  const btn = $("#btn-theme");
  if (btn) {
    btn.textContent = theme === "light" ? "🌙 深色主题" : "☀️ 浅色主题";
    btn.dataset.tip = theme === "light" ? "当前为浅色主题，点击切换为深色" : "当前为深色主题，点击切换为浅色";
  }
}
function initTheme() {
  const saved = localStorage.getItem("moss_theme") || "light";
  applyTheme(saved);
  const btn = $("#btn-theme");
  if (btn) {
    btn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "light";
      applyTheme(cur === "light" ? "dark" : "light");
    });
  }
}

// -------------------------------------------------------------------- init
(async function init() {
  initTheme();
  state.schema = await api("/api/schema");
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("moss_params") || "{}"); } catch (e) { /* ignore */ }
  setParams(saved);
  const out = $("#console");
  out.append(el("div", { class: "log-sys" }, ">>> [系统就绪] KohakuMoss WebUI 已连接。填写参数后点击 Train；先点 Scan Dataset 可预览每 epoch 的步数与分桶。"));
  loadPresets();
  showProbe();
  poll();
})();
