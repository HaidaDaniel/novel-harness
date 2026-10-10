"use strict";

const state = {
  runId: null,
  runGeneration: 0,
  sampleRunId: null,
  samplePreset: null,
  sampleNovelId: null,
  sampleWholeRunId: null,
  sampleWholeBusy: false,
  chapter: null,
  chapterGeneration: 0,
  poll: null,
  runners: [],
  presets: [],
  defaultPreset: null,
  "create-preset": null,
  "sample-preset": null,
  wizardNovels: [],
  wizardOptions: null,
  wizardBusy: false,
  wizardSample: null,
  wizardWholeRunId: null,
  wizardPoll: null,
};

const el = (id) => document.getElementById(id);

function setNotice(message, kind) {
  const node = el("notice");
  node.textContent = message || "";
  node.style.color = kind === "fail" ? "var(--fail)" : kind === "ok" ? "var(--ok)" : "var(--warn)";
}

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (error) {
      body = text;
    }
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : body || response.statusText;
    throw new Error(`${response.status}: ${detail}`);
  }
  return body;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(value, suffix = "") {
  if (value === null || value === undefined) return "unknown";
  return `${value}${suffix}`;
}

function formatTimestamp(value) {
  if (!value) return "unknown";
  return String(value)
    .replace("T", " ")
    .replace(/\.\d+(?=[+-]\d\d:\d\d$)/, "")
    .replace("+00:00", " UTC");
}

function statusClass(status) {
  if (["completed", "checks_passed"].includes(status)) return "ok";
  if (["failed", "blocked", "incomplete", "interrupted", "missing", "invalid", "conflict"].includes(status)) return "fail";
  if (["running", "waiting_retry", "paused", "needs_review"].includes(status)) return "warn";
  return "";
}

function badge(text, cls) {
  return `<span class="badge ${cls || ""}">${escapeHtml(text)}</span>`;
}

/* ---------------------------------------------------------------- tabs */

function showTab(name) {
  for (const section of document.querySelectorAll(".tab")) {
    section.classList.toggle("hidden", section.id !== `tab-${name}`);
  }
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  if (name !== "run" && state.poll) {
    clearInterval(state.poll);
    state.poll = null;
  }
  if (name === "bible") {
    loadBible();
  }
  if (name === "tm") {
    loadTranslationMemory();
  }
}

document.querySelectorAll("#tabs button").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
});

/* ------------------------------------------------------------- library */

async function loadRunners() {
  const select = el("create-runner");
  try {
    state.runners = await api("/runners");
    const options = state.runners
      .map((runner) => {
        const label = `${runner.name} (${runner.type}/${runner.model || "no model"})`;
        const disabled = runner.available ? "" : " disabled";
        return `<option value="${escapeHtml(runner.name)}"${disabled}>${escapeHtml(label)}</option>`;
      })
      .join("");
    select.innerHTML = options;
    el("sample-runner").innerHTML = options;
    const unavailable = state.runners.filter((runner) => !runner.available);
    el("runners-note").textContent = unavailable.length
      ? `Unavailable runners: ${unavailable.map((r) => `${r.name} (${r.reason})`).join("; ")}.`
      : "All configured runners report available (configuration check only; endpoint not probed).";
  } catch (error) {
    setNotice(error.message, "fail");
  }
}

/* --------------------------------------------------------------- presets */

function presetById(id) {
  return state.presets.find((preset) => preset.id === id) || null;
}

function selectedPreset(radioName) {
  const checked = document.querySelector(`input[name="${radioName}"]:checked`);
  return checked ? checked.value : null;
}

function presetCard(preset, radioName) {
  const disabled = preset.available ? "" : " disabled";
  const badges = [
    preset.recommended ? badge("Recommended", "ok") : "",
    preset.default ? badge("Default", "") : "",
  ].join(" ");
  const detail = preset.available
    ? `${escapeHtml(preset.description)}<br>` +
      `<span class="muted mono">${escapeHtml(preset.runner)} / ${escapeHtml(preset.pipeline)}` +
      `${preset.model ? " / " + escapeHtml(preset.model) : ""}</span>`
    : `<span class="warning">Unavailable: ${escapeHtml(preset.reason || "unavailable")}</span>`;
  return `<label class="preset-card${preset.available ? "" : " disabled"}">
    <input type="radio" name="${radioName}" value="${escapeHtml(preset.id)}"${disabled}>
    <span class="preset-title">${escapeHtml(preset.label)} ${badges}</span>
    <span class="preset-desc">${detail}</span>
  </label>`;
}

function customPresetCard(radioName) {
  const chosen = state[radioName];
  const checked = chosen === "custom" ? " checked" : "";
  return `<label class="preset-card">
    <input type="radio" name="${radioName}" value="custom"${checked}>
    <span class="preset-title">Custom / Advanced</span>
    <span class="preset-desc">Choose runner/pipeline manually (research / compatibility)</span>
  </label>`;
}

function renderPresetCards(containerId, radioName) {
  const container = el(containerId);
  container.innerHTML =
    state.presets.map((preset) => presetCard(preset, radioName)).join("") +
    customPresetCard(radioName);
  // One checked radio per group. The configured default preset wins only when it is
  // actually available; an unavailable default falls back to Custom with a warning
  // and never silently selects another quality preset.
  const chosen = resolveInitialPreset(radioName);
  const target = container.querySelector(`input[value="${chosen}"]`);
  if (target) target.checked = true;
  container.querySelectorAll('input[type="radio"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state[radioName] = radio.value;
      updatePresetNote(radioName);
    });
  });
}

function resolveInitialPreset(radioName) {
  if (state[radioName]) return state[radioName];
  const requested = state.defaultPreset;
  if (requested) {
    const preset = presetById(requested);
    if (preset && preset.available) return requested;
  }
  return "custom";
}

function defaultUnavailableWarning() {
  const requested = state.defaultPreset;
  if (!requested) return "";
  const preset = presetById(requested);
  if (!preset || preset.available) return "";
  return (
    `<span class="badge fail">Default preset unavailable</span> ` +
    `Default preset "${escapeHtml(preset.label)}" is unavailable: ` +
    `${escapeHtml(preset.reason || "unavailable")}. ` +
    "Configure it or choose another available mode."
  );
}

function updatePresetNote(radioName) {
  const kind = radioName.replace("-preset", "");
  const value = selectedPreset(radioName);
  const preset = presetById(value);
  const note = el(`${kind}-preset-note`);
  const advanced = el(`${kind}-advanced`);
  const warning = defaultUnavailableWarning();
  if (!preset) {
    note.innerHTML =
      (warning ? `<div class="warning fail">${warning}</div>` : "") +
      '<span class="badge warn">Custom</span> Manual runner/pipeline configuration is open below.';
    if (advanced) advanced.open = true;
    return;
  }
  const mode = preset.cloud
    ? '<span class="badge warn">External model</span> May consume subscription/API quota. ' +
      "Measured monetary cost: unknown."
    : '<span class="badge ok">Local model</span> No external translation provider required.';
  note.innerHTML = (warning ? `<div class="warning fail">${warning}</div>` : "") + mode;
  if (advanced) advanced.open = false;
}

async function loadPresets() {
  try {
    const data = await api("/presets");
    state.presets = data.presets || [];
    state.defaultPreset = data.default || null;
    renderPresetCards("create-presets", "create-preset");
    renderPresetCards("sample-presets", "sample-preset");
    updatePresetNote("create-preset");
    updatePresetNote("sample-preset");
    if (!state.presets.length) {
      el("create-preset-note").textContent =
        "No translation presets are configured; use Custom / Advanced below.";
      el("sample-preset-note").textContent =
        "No translation presets are configured; use Custom / Advanced below.";
    }
  } catch (error) {
    setNotice(error.message, "fail");
  }
}

function selectionPayload(radioName, runnerSelectId, pipelineSelectId) {
  const value = selectedPreset(radioName);
  if (value && value !== "custom") {
    return { preset: value };
  }
  return {
    runner: el(runnerSelectId).value || null,
    pipeline: el(pipelineSelectId).value || "direct",
  };
}

/* ---------------------------------------------------------- guided wizard */

function wizardMode() {
  const id = el("wizard-preset").value;
  if (id === "custom") return { id, pipeline: "direct" };
  return (state.wizardOptions?.presets || []).find((preset) => preset.id === id) || null;
}

function wizardProfile() {
  return (state.wizardOptions?.profiles || []).find(
    (profile) => profile.id === el("wizard-profile").value
  ) || null;
}

function wizardSelection() {
  const novelId = el("wizard-novel").value;
  const novel = state.wizardNovels.find(
    (item) => item.novel_id === novelId && item.status === "ready"
  );
  const mode = wizardMode();
  const profile = wizardProfile();
  if (!novel || !mode || !profile || !profile.available || mode.available === false) return null;
  if (mode.id !== "custom" && mode.execution_profile_id !== profile.id) return null;
  return { novelId, modeId: mode.id, profileId: profile.id, profile };
}

function wizardSelectionMatchesSample() {
  const selection = wizardSelection();
  const sample = state.wizardSample;
  return !!(selection && sample && selection.novelId === sample.novelId &&
    selection.modeId === sample.modeId && selection.profileId === sample.profileId);
}

function describeModel(value) {
  if (!value || value.status === "unknown") return "unknown";
  if (value.id) return value.id;
  const name = [value.provider, value.model].filter(Boolean).join("/");
  return `${name || "unknown"}${value.variant ? `#${value.variant}` : ""}`;
}

function renderWizardDetails() {
  const mode = wizardMode();
  const profile = wizardProfile();
  const note = el("wizard-profile-note");
  const details = el("wizard-execution");
  if (!profile) {
    note.textContent = mode ? "No compatible available connection. Configure one before starting." :
      "Choose a translation mode to see its connection.";
    details.innerHTML = "";
    return;
  }
  const source = profile.model_source || "unknown source";
  note.textContent = profile.available
    ? `${profile.harness} · ${describeModel(profile.resolved_model)} · ${source}. Cost: unknown.`
    : `Unavailable: ${profile.reason || "connection unavailable"}`;
  const rows = [
    ["Harness", profile.harness || "unknown"],
    ["Agent", profile.agent_id || "none"],
    ["Requested policy", profile.model_policy || "explicit"],
    ["Requested model", describeModel(profile.requested_model)],
    ["Resolved model", describeModel(profile.resolved_model)],
    ["Model source", source],
    ["Override", profile.model_policy === "override" ? "explicit configured override" : "none"],
    ["Pipeline", mode?.pipeline || "direct"],
    ["Stages", (mode?.stages || []).map((stage) =>
      `${stage.name}: ${stage.profile_id || "unconfigured"}`).join("; ") || "direct"],
  ];
  details.innerHTML = rows.map(([label, value]) =>
    `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`
  ).join("");
}

function renderWizardControls() {
  const selection = wizardSelection();
  const sample = state.wizardSample;
  const matching = wizardSelectionMatchesSample();
  el("wizard-start-sample").disabled = state.wizardBusy || !selection;
  el("wizard-open-sample").disabled = !sample;
  el("wizard-approve").disabled = state.wizardBusy || !matching ||
    sample.status !== "completed" || !sample.generationComplete || !!sample.approval;
  el("wizard-whole-book").disabled = state.wizardBusy || !matching ||
    !sample.approval || !!state.wizardWholeRunId;
  if (sample && !matching) {
    el("wizard-status").textContent =
      `Sample ${sample.runId} belongs to another book, mode or profile. Start a new sample.`;
  }
  renderWizardDetails();
}

function renderWizardProfiles() {
  const select = el("wizard-profile");
  const previous = select.value;
  const mode = wizardMode();
  const all = state.wizardOptions?.profiles || [];
  const profiles = mode?.id === "custom"
    ? all : all.filter((profile) => profile.id === mode?.execution_profile_id);
  select.innerHTML = '<option value="">Choose a connection</option>' + profiles.map((profile) =>
    `<option value="${escapeHtml(profile.id)}"${profile.available ? "" : " disabled"}>` +
    `${escapeHtml(profile.id)}${profile.available ? "" : ` — unavailable: ${escapeHtml(profile.reason || "unknown")}`}</option>`
  ).join("");
  if (mode?.id !== "custom" && profiles.length === 1 && profiles[0].available) {
    select.value = profiles[0].id;
  } else if (profiles.some((profile) => profile.id === previous && profile.available)) {
    select.value = previous;
  }
  select.disabled = mode?.id !== "custom";
  renderWizardControls();
}

function renderWizardModes() {
  const select = el("wizard-preset");
  const previous = select.value;
  const options = state.wizardOptions;
  if (!options) return;
  select.innerHTML = '<option value="">Choose a translation mode</option>' +
    options.presets.map((preset) =>
      `<option value="${escapeHtml(preset.id)}"${preset.available ? "" : " disabled"}>` +
      `${escapeHtml(preset.label)}${preset.available ? "" : ` — unavailable: ${escapeHtml(preset.reason || "unknown")}`}</option>`
    ).join("") + '<option value="custom">Custom / Advanced</option>';
  const defaultMode = options.presets.find(
    (preset) => preset.id === options.default && preset.available
  );
  if (previous === "custom" || options.presets.some((preset) =>
    preset.id === previous && preset.available)) select.value = previous;
  else if (defaultMode) select.value = defaultMode.id;
  renderWizardProfiles();
}

function renderWizardBooks() {
  const select = el("wizard-novel");
  const previous = select.value;
  const choices = state.wizardNovels.filter((novel) => novel.status === "ready");
  select.innerHTML = '<option value="">Choose a book</option>' + choices.map((novel) =>
    `<option value="${escapeHtml(novel.novel_id)}">${escapeHtml(novel.title)}</option>`
  ).join("");
  if (choices.some((novel) => novel.novel_id === previous)) select.value = previous;
  renderWizardControls();
}

async function loadWizardOptions() {
  try {
    state.wizardOptions = await api("/wizard/options");
    renderWizardModes();
  } catch (error) {
    el("wizard-status").textContent = `Connections unavailable: ${error.message}`;
    renderWizardControls();
  }
}

function wizardPayload(selection) {
  if (selection.modeId !== "custom") return { novel_id: selection.novelId, preset: selection.modeId };
  const field = selection.profile.configured_profile ? "execution_profile" : "runner";
  return { novel_id: selection.novelId, [field]: selection.profileId, pipeline: "direct" };
}

async function refreshWizardSample() {
  const sample = state.wizardSample;
  if (!sample) return;
  try {
    const [status, approval] = await Promise.all([
      api(`/runs/${encodeURIComponent(sample.runId)}/status`),
      api(`/sample-runs/${encodeURIComponent(sample.runId)}/approval`),
    ]);
    if (state.wizardSample !== sample) return;
    sample.status = status.status;
    sample.generationComplete = status.generation_complete;
    sample.approval = approval;
    if (["completed", "failed", "blocked", "interrupted"].includes(status.status) && state.wizardPoll) {
      clearInterval(state.wizardPoll);
      state.wizardPoll = null;
    }
    el("wizard-status").textContent = approval
      ? `Sample ${sample.runId} approved. Ready for a linked whole-book run.`
      : `Sample ${sample.runId}: ${status.status}${status.error ? ` — ${status.error}` : ""}.`;
    renderWizardControls();
    return true;
  } catch (error) {
    el("wizard-status").textContent = `Sample status unavailable: ${error.message}`;
    setNotice(error.message, "fail");
    return false;
  }
}

el("wizard-novel").addEventListener("change", renderWizardControls);
el("wizard-preset").addEventListener("change", renderWizardProfiles);
el("wizard-profile").addEventListener("change", renderWizardControls);

el("wizard-start-sample").addEventListener("click", async () => {
  const selection = wizardSelection();
  if (!selection || state.wizardBusy) return;
  state.wizardBusy = true;
  let createdRunId = null;
  let workerStarted = false;
  renderWizardControls();
  try {
    const previousRunId = state.wizardSample?.runId;
    const created = await api("/sample-runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(wizardPayload(selection)),
    });
    createdRunId = created.run_id;
    state.wizardSample = {
      runId: created.run_id, novelId: selection.novelId,
      modeId: selection.modeId, profileId: selection.profileId,
      status: "created", generationComplete: false, approval: null,
    };
    if (created.run_id !== previousRunId) state.wizardWholeRunId = null;
    state.sampleRunId = created.run_id;
    state.sampleNovelId = selection.novelId;
    state.samplePreset = selection.modeId === "custom" ? null : selection.modeId;
    el("wizard-status").textContent = `Sample ${created.run_id} created; starting worker…`;
    await api(`/runs/${encodeURIComponent(created.run_id)}/start`, { method: "POST" });
    workerStarted = true;
    if (!await refreshWizardSample()) throw new Error("Sample status could not be refreshed.");
    if (state.wizardPoll) clearInterval(state.wizardPoll);
    state.wizardPoll = setInterval(refreshWizardSample, 3000);
    await loadLibrary();
    openRun(created.run_id);
  } catch (error) {
    const existing = createdRunId && !workerStarted && await refreshWizardSample();
    el("wizard-status").textContent = existing && state.wizardSample.status === "completed"
      ? `Sample ${createdRunId} was already completed. Review it and use its existing approval.`
      : existing && state.wizardSample.status === "running"
        ? `Sample ${createdRunId} is already running. Open it to monitor progress.`
        : createdRunId
          ? workerStarted
            ? `Sample ${createdRunId} started, but status refresh failed: ${error.message}. Open the run to inspect it.`
            : `Sample ${createdRunId} was created, but its worker did not start: ${error.message}. Open the run to retry.`
          : `Sample could not be created: ${error.message}`;
    setNotice(error.message, "fail");
  } finally {
    state.wizardBusy = false;
    renderWizardControls();
  }
});

el("wizard-open-sample").addEventListener("click", () => {
  if (state.wizardSample) openRun(state.wizardSample.runId);
});

el("wizard-approve").addEventListener("click", async () => {
  if (!wizardSelectionMatchesSample() || state.wizardBusy) return;
  state.wizardBusy = true;
  renderWizardControls();
  try {
    if (!await refreshWizardSample()) throw new Error("Sample status could not be refreshed.");
    if (state.wizardSample.status !== "completed" || !state.wizardSample.generationComplete) {
      throw new Error("Sample must complete successfully before approval.");
    }
    await api(`/sample-runs/${encodeURIComponent(state.wizardSample.runId)}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "ui" }),
    });
    if (!await refreshWizardSample()) throw new Error("Sample approval could not be refreshed.");
  } catch (error) {
    el("wizard-status").textContent = `Approval refused: ${error.message}`;
    setNotice(error.message, "fail");
  } finally {
    state.wizardBusy = false;
    renderWizardControls();
  }
});

el("wizard-whole-book").addEventListener("click", async () => {
  if (!wizardSelectionMatchesSample() || state.wizardBusy || state.wizardWholeRunId) return;
  state.wizardBusy = true;
  renderWizardControls();
  try {
    if (!await refreshWizardSample()) throw new Error("Sample approval could not be refreshed.");
    const sample = state.wizardSample;
    if (!sample.approval || !wizardSelectionMatchesSample()) {
      throw new Error("Approve this completed sample before whole-book launch.");
    }
    const payload = {
      sample_run_id: sample.runId, approved: true, novel_id: sample.novelId,
    };
    if (sample.modeId !== "custom") payload.preset = sample.modeId;
    const created = await api("/whole-book-runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.wizardWholeRunId = created.run_id;
    el("wizard-status").textContent = `Whole-book run ${created.run_id} created; starting worker…`;
    await api(`/runs/${encodeURIComponent(created.run_id)}/start`, { method: "POST" });
    el("wizard-status").textContent = `Whole-book run ${created.run_id} started.`;
    await loadLibrary();
    openRun(created.run_id);
  } catch (error) {
    el("wizard-status").textContent = state.wizardWholeRunId
      ? `Whole-book run ${state.wizardWholeRunId} was created, but its worker did not start: ${error.message}. Open the run to retry.`
      : `Whole-book launch refused: ${error.message}`;
    setNotice(error.message, "fail");
    if (state.wizardWholeRunId) {
      await loadLibrary();
      openRun(state.wizardWholeRunId);
    }
  } finally {
    state.wizardBusy = false;
    renderWizardControls();
  }
});

async function loadLibrary() {
  el("novel-cards").innerHTML = `<p class="muted">Loading source novels…</p>`;
  try {
    const library = await api("/library");
    const novels = library.novels || [];
    state.wizardNovels = novels;
    renderWizardBooks();
    const choices = novels.filter((novel) => novel.status === "ready");
    for (const id of ["create-novel", "sample-novel"]) {
      const select = el(id);
      const previous = select.value;
      select.innerHTML = choices.length
        ? choices.map((novel) => `<option value="${escapeHtml(novel.novel_id)}">${escapeHtml(novel.title)}</option>`).join("")
        : `<option value="">No selectable books</option>`;
      if (choices.some((novel) => novel.novel_id === previous)) select.value = previous;
    }
    el("library-diagnostics").innerHTML = (library.diagnostics || [])
      .map((message) => `<div class="warning">${escapeHtml(message)}</div>`).join("");
    el("novel-cards").innerHTML = novels.length ? novels.map((novel) => {
      const run = novel.active_run || novel.latest_run;
      const provider = novel.source_type ? `${novel.source_type}:${novel.source_id}` : "local source";
      const progress = Math.round((novel.progress || 0) * 100);
      const final = novel.final_layer;
      const selectable = novel.status === "ready";
      return `<article class="novel-card">
        <h3>${escapeHtml(novel.title)}</h3>
        <p>${badge(novel.status, statusClass(novel.status))} ${escapeHtml(provider)}</p>
        <p>${novel.chapter_count} chapter(s) · ${escapeHtml(novel.translation_state)}</p>
        <p class="muted">${run ? `Machine: ${progress}% · Final: ${final.accepted}/${final.total_chunks} accepted · ${final.needs_review} review · ${final.stale} stale · ${final.machine_only} machine-only` : "No runs yet"}</p>
        <p class="muted">${run ? `Run: ${escapeHtml(run.run_id)} (${escapeHtml(run.status)})` : ""}</p>
        ${novel.error ? `<p class="warning fail">${escapeHtml(novel.error)}</p>` : ""}
        <div class="controls">
          <button data-translate="${escapeHtml(novel.novel_id || "")}" ${selectable ? "" : "disabled"}>Translate</button>
          <button data-continue="${escapeHtml(novel.continue_run_id || "")}" ${novel.continue_run_id ? "" : "disabled"}>Continue</button>
          <button data-novel-open="${escapeHtml(novel.novel_id || "")}" ${novel.novel_id ? "" : "disabled"}>Open</button>
        </div>
      </article>`;
    }).join("") : `<p class="muted">No source novels found. Add chapter folders under a configured root and rescan.</p>`;
    el("novel-cards").querySelectorAll("button[data-translate]").forEach((button) => {
      button.addEventListener("click", () => {
        el("create-novel").value = button.dataset.translate;
        el("sample-novel").value = button.dataset.translate;
        el("wizard-novel").value = button.dataset.translate;
        renderWizardControls();
        el("wizard-novel").scrollIntoView({ behavior: "smooth" });
      });
    });
    el("novel-cards").querySelectorAll("button[data-continue]").forEach((button) => {
      button.addEventListener("click", () => openRun(button.dataset.continue));
    });
    el("novel-cards").querySelectorAll("button[data-novel-open]").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          const novel = await api(`/library/${encodeURIComponent(button.dataset.novelOpen)}`);
          el("novel-detail").textContent = `${novel.title} · ${novel.source_path} · chapters: ${novel.chapters.map((chapter) => chapter.number).join(", ")}`;
        } catch (error) {
          el("novel-detail").textContent = error.message;
          setNotice(error.message, "fail");
        }
      });
    });
    const data = await api("/runs");
    el("library-source").textContent =
      `${novels.length} source item(s) indexed; ${(data.runs || []).length} persisted run(s).`;
    const rows = data.runs || [];
    const body = document.querySelector("#runs-table tbody");
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="6" class="muted">No runs yet. Create one above.</td></tr>`;
      return;
    }
    body.innerHTML = rows
      .map((run) => {
        const chunks = run.chunks || {};
        const progress = Math.round((run.progress || 0) * 100);
        return `<tr>
          <td class="mono">${escapeHtml(run.run_id)}</td>
          <td>${badge(run.status, statusClass(run.status))}</td>
          <td>${escapeHtml(run.workflow)}</td>
          <td>
            <div class="progress-bar"><span style="width:${progress}%"></span></div>
            ${progress}% (${chunks.completed || 0}/${run.total_chunks || 0})
          </td>
          <td class="muted">ok ${chunks.completed || 0} / pend ${chunks.pending || 0} /
              fail ${chunks.failed || 0} / run ${chunks.running || 0}</td>
          <td><button data-open="${escapeHtml(run.run_id)}">Open</button></td>
        </tr>`;
      })
      .join("");
    body.querySelectorAll("button[data-open]").forEach((button) => {
      button.addEventListener("click", () => openRun(button.dataset.open));
    });
  } catch (error) {
    state.wizardNovels = [];
    renderWizardBooks();
    el("novel-cards").innerHTML = `<p class="muted">Library unavailable: ${escapeHtml(error.message)}</p>`;
    setNotice(error.message, "fail");
  }
}

el("create-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const chaptersRaw = el("create-chapters").value.trim();
  const chapters = chaptersRaw
    ? chaptersRaw.split(",").map((value) => parseInt(value.trim(), 10)).filter((n) => Number.isFinite(n))
    : null;
  const chunkCharsRaw = el("create-chunk-chars").value.trim();
  const payload = {
    ...selectionPayload("create-preset", "create-runner", "create-pipeline"),
    novel_id: el("create-novel").value || null,
    chapters,
    chunk_chars: chunkCharsRaw ? parseInt(chunkCharsRaw, 10) : null,
  };
  try {
    const created = await api("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setNotice(`Created run ${created.run_id}`, "ok");
    await loadLibrary();
    openRun(created.run_id);
  } catch (error) {
    setNotice(error.message, "fail");
  }
});

el("refresh-library").addEventListener("click", async () => {
  try {
    await api("/library/rescan", { method: "POST" });
    await loadLibrary();
  } catch (error) {
    setNotice(error.message, "fail");
  }
});

/* ----------------------------------------------------------------- run */

async function control(path) {
  if (!state.runId) return;
  const generation = state.runGeneration;
  try {
    const result = await api(path, { method: "POST" });
    if (generation !== state.runGeneration) return;
    setNotice(`${path}: ${JSON.stringify(result)}`, "ok");
    await renderRun();
  } catch (error) {
    if (generation !== state.runGeneration) return;
    setNotice(error.message, "fail");
  }
}

function clearChapterView(resetNumber = true) {
  el("chapter-source").textContent = "";
  el("chapter-translation").textContent = "";
  el("chapter-warnings").innerHTML = "";
  el("review-desk").innerHTML = "";
  el("chapter-chunks").innerHTML = "";
  const number = el("chapter-number");
  if (number && resetNumber) number.value = "";
}

function openRun(runId) {
  // Bumping the generation invalidates any in-flight response for a run the
  // user has since switched away from.
  state.runGeneration += 1;
  state.runId = runId;
  state.chapter = null;
  el("run-id-label").textContent = runId;
  clearChapterView();
  showTab("run");
  renderRun();
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(renderRun, 3000);
}

async function renderRun() {
  if (!state.runId) return;
  const runId = state.runId;
  const generation = state.runGeneration;
  try {
    const [status, progress, metrics] = await Promise.all([
      api(`/runs/${runId}/status`),
      api(`/runs/${runId}/progress`).catch(() => null),
      api(`/runs/${runId}/metrics`).catch(() => null),
    ]);
    if (runId !== state.runId || generation !== state.runGeneration) return;
    renderRunSummary(status);
    renderRunProgress(progress, status);
    renderRunUsage(metrics);
    renderRunChapters(status);
    setNotice("");
  } catch (error) {
    if (runId !== state.runId || generation !== state.runGeneration) return;
    setNotice(error.message, "fail");
  }
}

function renderRunSummary(status) {
  const runners = (status.runners || [])
    .map((runner) => `${runner.stage}: ${runner.name} / ${runner.model}`)
    .join("<br>");
  el("run-summary").innerHTML = `
    <div class="kv">
      <div><dt>Status</dt><dd>${badge(status.status, statusClass(status.status))}</dd></div>
      <div><dt>Workflow</dt><dd>${escapeHtml(status.workflow)}</dd></div>
      <div><dt>Generation complete</dt><dd>${escapeHtml(status.generation_complete)}</dd></div>
      <div><dt>Chapters</dt><dd>${(status.chapters || []).length}</dd></div>
      <div><dt>Created</dt><dd>${escapeHtml(formatTimestamp(status.created_at))}</dd></div>
      <div><dt>Updated</dt><dd>${escapeHtml(formatTimestamp(status.updated_at))}</dd></div>
      <div><dt>Runners</dt><dd>${runners || "unknown"}</dd></div>
      <div><dt>Error</dt><dd>${status.error ? escapeHtml(status.error) : "-"}</dd></div>
    </div>`;
}

function renderRunProgress(progress, status) {
  const chunks = status.chunks || {};
  const percent = Math.round((status.progress || 0) * 100);
  let current = "-";
  if (progress && progress.current) {
    current = `chapter ${progress.current.chapter}, chunk ${progress.current.chunk}` +
      (progress.current.stage ? `, stage ${progress.current.stage}` : "");
  }
  const eta = progress ? progress.eta_seconds : null;
  const throughput = progress ? progress.throughput_chunks_per_minute : null;
  const retry = status.retry ? JSON.stringify(status.retry) : "-";
  el("run-progress").innerHTML = `
    <h3>Progress</h3>
    <div class="progress-bar"><span style="width:${percent}%"></span></div>
    <div class="kv">
      <div><dt>Completed</dt><dd>${chunks.completed || 0}</dd></div>
      <div><dt>Pending</dt><dd>${chunks.pending || 0}</dd></div>
      <div><dt>Failed</dt><dd>${chunks.failed || 0}</dd></div>
      <div><dt>Running</dt><dd>${chunks.running || 0}</dd></div>
      <div><dt>Corrupt</dt><dd>${chunks.corrupt || 0}</dd></div>
      <div><dt>Current</dt><dd>${escapeHtml(current)}</dd></div>
      <div><dt>Elapsed</dt><dd>${fmt(progress ? progress.elapsed_seconds : null, " s")}</dd></div>
      <div><dt>Throughput</dt><dd>${fmt(throughput, " chunks/min")}</dd></div>
      <div><dt>ETA</dt><dd>${fmt(eta, " s")}</dd></div>
      <div><dt>Retry</dt><dd>${escapeHtml(retry)}</dd></div>
    </div>`;
}

function renderRunUsage(metrics) {
  if (!metrics || !(metrics.stages || []).length) {
    el("run-usage").innerHTML = `<h3>Token usage</h3><p class="muted">No attempts recorded yet.</p>`;
    return;
  }
  const rows = metrics.stages
    .map((stage) => {
      const usage = stage.usage || {};
      const total =
        usage.prompt_tokens == null && usage.completion_tokens == null
          ? "unknown"
          : `${usage.prompt_tokens == null ? "?" : usage.prompt_tokens} + ` +
            `${usage.completion_tokens == null ? "?" : usage.completion_tokens}`;
      return `<tr>
        <td>${escapeHtml(stage.stage)}</td>
        <td>${escapeHtml(stage.runner)}</td>
        <td>${escapeHtml(stage.model)}</td>
        <td>${stage.attempts}</td>
        <td>${stage.validation_failures}</td>
        <td>${stage.recovery_exhausted}</td>
        <td>${escapeHtml(total)}</td>
      </tr>`;
    })
    .join("");
  el("run-usage").innerHTML = `
    <h3>Token usage</h3>
    <p class="muted">Unknown counters stay unknown; they are never zero-filled.</p>
    <table><thead><tr><th>Stage</th><th>Runner</th><th>Model</th><th>Attempts</th>
      <th>Validation fails</th><th>Exhausted</th><th>Prompt + completion</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
}

function renderRunChapters(status) {
  const chapters = status.chapters || [];
  if (!chapters.length) {
    el("run-chapters").innerHTML = `<p class="muted">No chapters recorded.</p>`;
    return;
  }
  el("run-chapters").innerHTML = `<div class="chunk-list">${chapters
    .map((number) => `<button class="badge" data-chapter="${number}">${number}</button>`)
    .join("")}</div>`;
  el("run-chapters").querySelectorAll("button[data-chapter]").forEach((button) => {
    button.addEventListener("click", () => {
      el("chapter-number").value = button.dataset.chapter;
      showTab("chapter");
      loadChapter();
    });
  });
}

el("btn-start").addEventListener("click", () => control(`/runs/${state.runId}/start`));
el("btn-pause").addEventListener("click", () => control(`/runs/${state.runId}/pause`));
el("btn-resume").addEventListener("click", () => control(`/runs/${state.runId}/resume`));
el("btn-retry").addEventListener("click", () => control(`/runs/${state.runId}/retry-failed`));
el("btn-build").addEventListener("click", () => control(`/runs/${state.runId}/build`));

/* ------------------------------------------------------------- chapter */

function isCurrentChapter(view) {
  return view && view.runId === state.runId && view.generation === state.runGeneration &&
    view.chapter === state.chapter && view.chapterGeneration === state.chapterGeneration;
}

async function loadChapter() {
  if (!state.runId) {
    setNotice("Open a run first.", "fail");
    return;
  }
  const chapter = parseInt(el("chapter-number").value, 10);
  if (!Number.isFinite(chapter)) {
    setNotice("Enter a chapter number.", "fail");
    return;
  }
  const view = {
    runId: state.runId,
    chapter,
    generation: state.runGeneration,
    chapterGeneration: ++state.chapterGeneration,
  };
  state.chapter = chapter;
  clearChapterView(false);
  try {
    const data = await api(`/runs/${view.runId}/chapters/${chapter}`);
    if (!isCurrentChapter(view)) return;
    renderChapter(data, view);
    setNotice("");
  } catch (error) {
    if (!isCurrentChapter(view)) return;
    setNotice(error.message, "fail");
  }
}

function renderChapter(data, view) {
  el("chapter-source").textContent = data.source || "";
  el("chapter-translation").textContent = data.translation || "";
  const warnings = [];
  if (!data.complete) {
    warnings.push(`<div class="warning fail">Chapter is incomplete.</div>`);
  }
  for (const entry of data.validation || []) {
    if (entry.level && entry.level !== "PASS") {
      const issues = (entry.issues || []).join(", ") || entry.error || "";
      warnings.push(
        `<div class="warning ${entry.level === "FAIL" ? "fail" : ""}">` +
          `chunk ${entry.ordinal} [${escapeHtml(entry.level)}] ${escapeHtml(issues)}</div>`
      );
    }
    if (entry.status === "failed") {
      warnings.push(`<div class="warning fail">chunk ${entry.ordinal} failed: ${escapeHtml(entry.error || "")}</div>`);
    }
  }
  el("chapter-warnings").innerHTML = warnings.join("");
  el("review-desk").innerHTML = renderReviewDesk(data);
  attachReviewActions(view);
  el("chapter-chunks").innerHTML = el("toggle-chunks").checked
    ? `<div class="chunk-list">${(data.validation || [])
        .map((entry) => badge(`#${entry.ordinal} ${entry.status}`, statusClass(entry.status)))
        .join("")}</div>`
    : "";
}

function renderReviewDesk(data) {
  const layers = data.final_status || [];
  if (!layers.length) return "";
  const rows = layers.map((chunk) => {
    const stale = chunk.stale ? badge("stale machine", "fail") : "";
    const machineHash = (chunk.machine_sha256 || "").slice(0, 12);
    return `<div class="review-item" data-chunk="${escapeHtml(chunk.chunk_id)}"
        data-machine="${escapeHtml(chunk.machine_sha256 || "")}"
        data-final-sha="${escapeHtml(chunk.final_sha256 || "")}">
      <div class="review-head">chunk ${chunk.ordinal} ${badge(chunk.status, statusClass(chunk.status))}
        ${stale}<span class="muted mono">machine ${escapeHtml(machineHash)}</span></div>
      <div class="review-body">
        <div><h4>Machine candidate</h4><pre>${escapeHtml(chunk.machine || "")}</pre></div>
        <div><h4>Final layer</h4>
          <textarea data-role="final" aria-label="Final layer for chunk ${escapeHtml(chunk.ordinal)}" rows="5">${escapeHtml(chunk.final || "")}</textarea>
          <div class="controls">
            <button data-act="edit">Save edit</button>
            <button data-act="accept">Accept</button>
            <button data-act="needs_review">Needs review</button>
            <button data-act="revert">Revert to machine</button>
            <button data-act="requeue">Requeue</button>
          </div>
        </div>
      </div>
    </div>`;
  });
  return `<h3>Review desk</h3>` +
    `<p class="muted">source \u2192 machine candidate \u2192 user accepted/final. The machine candidate is never overwritten.</p>` +
    rows.join("");
}

function attachReviewActions(view) {
  document.querySelectorAll("#review-desk .review-item").forEach((item) => {
    const chunkId = item.dataset.chunk;
    const machine = item.dataset.machine;
    const finalSha = item.dataset.finalSha || null;
    const textarea = item.querySelector('textarea[data-role="final"]');
    item.querySelectorAll("button[data-act]").forEach((button) => {
      button.addEventListener("click", () =>
        reviewAction(button.dataset.act, chunkId, machine, finalSha, textarea, view)
      );
    });
  });
}

async function reviewAction(action, chunkId, machine, finalSha, textarea, view) {
  // A detached or queued old button keeps its rendered identity; it cannot
  // borrow a newly selected chapter, even when chunk hashes happen to match.
  if (!isCurrentChapter(view)) return;
  const {runId, chapter} = view;
  const base = `/runs/${runId}/chapters/${chapter}/chunks/${chunkId}`;
  try {
    if (action === "requeue") {
      await api(`${base}/requeue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_machine_sha256: machine }),
      });
    } else {
      const payload = { action, expected_machine_sha256: machine, actor: "user" };
      if (finalSha) payload.expected_final_sha256 = finalSha;
      if (action === "edit" || action === "needs_review") payload.text = textarea.value;
      await api(`${base}/final`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }
    if (!isCurrentChapter(view)) return;
    setNotice(`${action}: ok`, "ok");
    await loadChapter();
  } catch (error) {
    if (!isCurrentChapter(view)) return;
    setNotice(error.message, "fail");
  }
}

el("btn-load-chapter").addEventListener("click", loadChapter);
el("toggle-chunks").addEventListener("change", () => {
  if (!state.runId) return;
  loadChapter();
});

/* ------------------------------------------------------------ sample first */

function sampleChapters() {
  const raw = el("sample-chapters").value.trim();
  if (!raw) return null;
  return raw.split(",").map((token) => parseInt(token.trim(), 10)).filter((n) => !Number.isNaN(n));
}

async function createSampleFromForm(start) {
  const payload = {
    ...selectionPayload("sample-preset", "sample-runner", "sample-pipeline"),
    novel_id: el("sample-novel").value || null,
    chapters: sampleChapters(),
  };
  const previousRunId = state.sampleRunId;
  const created = await api("/sample-runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.sampleRunId = created.run_id;
  state.sampleNovelId = payload.novel_id;
  if (created.run_id !== previousRunId) state.sampleWholeRunId = null;
  const value = selectedPreset("sample-preset");
  state.samplePreset = value && value !== "custom" ? value : null;
  el("sample-run-label").textContent = `Sample run: ${created.run_id}`;
  await loadLibrary();
  if (start) {
    await api(`/runs/${encodeURIComponent(created.run_id)}/start`, { method: "POST" });
    el("sample-status").textContent = "Sample worker started. Review it, then approve.";
    openRun(created.run_id);
  } else {
    el("sample-status").textContent = "Sample created. Start it, then review it.";
  }
}

el("sample-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await createSampleFromForm(false);
  } catch (error) {
    setNotice(error.message, "fail");
  }
});

el("btn-sample-create-start").addEventListener("click", async () => {
  try {
    await createSampleFromForm(true);
  } catch (error) {
    setNotice(error.message, "fail");
  }
});

el("btn-sample-open").addEventListener("click", () => {
  if (!state.sampleRunId) return setNotice("Create a sample first", "fail");
  openRun(state.sampleRunId);
  showTab("run");
});

el("btn-sample-approve").addEventListener("click", async () => {
  if (!state.sampleRunId) return setNotice("Create a sample first", "fail");
  try {
    const approval = await api(
      `/sample-runs/${encodeURIComponent(state.sampleRunId)}/approve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor: "ui" }),
      }
    );
    el("sample-status").textContent = `Approved at ${approval.approved_at}.`;
  } catch (error) {
    setNotice(error.message, "fail");
  }
});

el("btn-whole-book").addEventListener("click", async () => {
  if (!state.sampleRunId) return setNotice("Approve a sample first", "fail");
  if (state.sampleWholeBusy || state.sampleWholeRunId) return;
  state.sampleWholeBusy = true;
  el("btn-whole-book").disabled = true;
  try {
    const approval = await api(`/sample-runs/${encodeURIComponent(state.sampleRunId)}/approval`);
    if (!approval) throw new Error("Approve the completed sample before whole-book launch.");
    // Whole-book inherits the approved sample's frozen settings; send the sample's
    // preset only so the server can verify it still matches. Never a different preset.
    const payload = { sample_run_id: state.sampleRunId, approved: true, novel_id: state.sampleNovelId };
    if (state.samplePreset) payload.preset = state.samplePreset;
    const created = await api("/whole-book-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.sampleWholeRunId = created.run_id;
    // Start the whole-book run through the persisted worker, never inline.
    await api(`/runs/${encodeURIComponent(created.run_id)}/start`, { method: "POST" });
    el("sample-status").textContent = `Whole-book run: ${created.run_id} (worker started).`;
    await loadLibrary();
    openRun(created.run_id);
    showTab("run");
  } catch (error) {
    setNotice(error.message, "fail");
  } finally {
    state.sampleWholeBusy = false;
    el("btn-whole-book").disabled = !!state.sampleWholeRunId;
  }
});

/* ------------------------------------------------------ translation memory */

async function loadTranslationMemory() {
  try {
    const data = await api("/translation-memory");
    el("tm-count").textContent = `${data.count} approved entr${data.count === 1 ? "y" : "ies"}.`;
    el("tm-table").querySelector("tbody").innerHTML = (data.entries || [])
      .map((entry) => {
        const p = entry.provenance || {};
        return `<tr>
          <td class="mono">${escapeHtml(entry.fingerprint.slice(0, 16))}…</td>
          <td>${escapeHtml(p.source_run_id || "unknown")}</td>
          <td>${escapeHtml(p.actor || "unknown")}</td>
          <td>${escapeHtml(p.approved_at || "unknown")}</td>
        </tr>`;
      })
      .join("");
    el("tm-empty").classList.toggle("hidden", (data.entries || []).length > 0);
  } catch (error) {
    setNotice(error.message, "fail");
  }
}

el("refresh-tm").addEventListener("click", loadTranslationMemory);

/* ------------------------------------------------------------ book bible */

function bibleRow(entry) {
  const aliases = (entry.aliases || []).map(escapeHtml).join(", ");
  return `<tr class="review-item" data-id="${escapeHtml(entry.entry_id)}">
    <td>${escapeHtml(entry.kind)}</td>
    <td>${escapeHtml(entry.source_form)}</td>
    <td contenteditable="true" data-field="target">${escapeHtml(entry.target_form)}</td>
    <td contenteditable="true" data-field="aliases">${aliases}</td>
    <td>${badge(entry.trust, statusClass(entry.trust))}</td>
    <td class="controls">
      <button data-act="confirm">Confirm</button>
      <button data-act="lock">Lock</button>
      <button data-act="edit">Save edit</button>
      <button data-act="reject">Reject</button>
    </td>
  </tr>`;
}

function renderBible(data) {
  el("bible-projection-hash").textContent = data.projection_sha256;
  const rows = data.entries || [];
  el("bible-table").querySelector("tbody").innerHTML = rows.map(bibleRow).join("");
  el("bible-empty").classList.toggle("hidden", rows.length > 0);
  attachBibleActions();
}

async function loadBible() {
  try {
    renderBible(await api("/book-bible"));
  } catch (error) {
    setNotice(error.message, "fail");
  }
}

function attachBibleActions() {
  document.querySelectorAll("#bible-table .review-item").forEach((row) => {
    const entryId = row.dataset.id;
    row.querySelectorAll("button[data-act]").forEach((button) => {
      button.addEventListener("click", async () => {
        const action = button.dataset.act;
        const payload = { action, actor: "ui" };
        if (action === "edit") {
          payload.target_form = row.querySelector('[data-field="target"]').textContent.trim();
          const raw = row.querySelector('[data-field="aliases"]').textContent.trim();
          payload.aliases = raw ? raw.split(",").map((a) => a.trim()).filter(Boolean) : [];
        }
        try {
          await api(`/book-bible/${encodeURIComponent(entryId)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          setNotice(`book bible ${action}: ok`, "ok");
          await loadBible();
        } catch (error) {
          setNotice(error.message, "fail");
        }
      });
    });
  });
}

el("refresh-bible").addEventListener("click", loadBible);
el("bible-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const split = (value) => value.split(",").map((a) => a.trim()).filter(Boolean);
  const entry = {
    kind: el("bible-kind").value,
    source_form: el("bible-source").value.trim(),
    target_form: el("bible-target").value.trim(),
    aliases: split(el("bible-aliases").value),
    source_anchors: split(el("bible-anchors").value),
    notes: el("bible-notes").value.trim() || null,
  };
  try {
    const result = await api("/book-bible", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries: [entry], proposer: "ui" }),
    });
    el("bible-note").textContent = `Created ${result.created.length}; skipped ${result.skipped.length}.`;
    el("bible-form").reset();
    await loadBible();
  } catch (error) {
    setNotice(error.message, "fail");
  }
});

/* ---------------------------------------------------------------- init */

loadRunners();
loadPresets();
loadWizardOptions();
loadLibrary();
showTab("library");
