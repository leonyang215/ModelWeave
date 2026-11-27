import {
  escapeHTML as e,
  count,
  percent,
  duration,
  workerStatus,
  filterWorkers,
  filterModels,
  kvRatio,
  mutationSpec,
  filterPlugins,
  pluginExample,
} from "./utils.mjs";

const $ = (id) => document.getElementById(id);
const roles = ["encode", "prefill", "decode", "unified"];
const roleNames = {
  encode: "多模态编码",
  prefill: "预填充",
  decode: "解码",
  unified: "一体化推理",
};
const pages = {
  overview: ["运行概览", "网关、模型与推理实例的实时运行视图。"],
  models: ["模型目录", "查看模型别名、执行模式与各角色的调度配置。"],
  workers: ["推理实例", "按模型、角色和状态筛选实例，安全地摘流或隔离。"],
  blacklist: [
    "实例黑名单",
    "隔离异常实例；规则按实例 ID 生效，不随重新注册消失。",
  ],
  tenants: ["租户配额", "只读查看租户限额、可用余额与全局并发占用。"],
  plugins: ["调度插件", "浏览可组合算法、参数与各模型实际生效的权重。"],
};
const statuses = {
  active: ["可调度", "green-tag"],
  blocked: ["已拉黑", "red-tag"],
  draining: ["摘流中", "amber-tag"],
  unhealthy: ["健康冷却", "red-tag"],
};
const actions = {
  block: [
    "拉黑推理实例",
    "确认拉黑",
    "停止为此实例分配新的请求。在途请求不受影响。",
  ],
  unblock: [
    "解除实例拉黑",
    "确认解除",
    "解除按 ID 设置的隔离规则，允许实例重新参与调度。",
  ],
  drain: [
    "摘流推理实例",
    "确认摘流",
    "暂停新请求，等待在途请求自然结束；适合短期维护。",
  ],
  resume: ["恢复实例调度", "确认恢复", "清除此注册实例的管理摘流标记。"],
};
const state = {
  page: "overview",
  data: null,
  connected: true,
  stale: false,
  loading: false,
  busy: false,
  updatedAt: null,
  error: "",
  modelQuery: "",
  blockQuery: "",
  pluginQuery: "",
  filters: { query: "", model: "", role: "", status: "" },
};
// Never persist this credential in localStorage, sessionStorage, cookies or URLs.
let token = "";
let generation = 0;
let inFlight = null;
let pendingAction = null;
let toastTimer;
let composing = false;

class APIError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json" };
  if (token) headers["X-ModelWeave-Token"] = token;
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    cache: "no-store",
    credentials: "omit",
    redirect: "error",
    signal: AbortSignal.timeout(10000),
    ...(options.body === undefined
      ? {}
      : { body: JSON.stringify(options.body) }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new APIError(
      response.status,
      body.message || body.error?.message || `HTTP ${response.status}`,
    );
  return body;
}

function errorMessage(err) {
  if (err.status === 401) return "管理令牌缺失、无效或已变更，请重新连接。";
  if (err.status === 503) return "网关状态后端暂不可用，请恢复后刷新。";
  if (err.name === "TimeoutError") return "请求超时，请检查网关状态后重试。";
  if (err instanceof TypeError) return "无法连接网关，请检查网络后重试。";
  return err.message || "请求失败，请重试。";
}

function showConnection(message = "") {
  $("connection-error").textContent = message;
  $("connection-error").hidden = !message;
  if (!$("connection-dialog").open) $("connection-dialog").showModal();
  $("admin-token").focus();
}

function clearConnection() {
  generation++;
  token = "";
  $("admin-token").value = "";
  state.connected = false;
  state.data = null;
  state.updatedAt = null;
  state.stale = false;
  state.error = "";
  state.loading = false;
  $("details-dialog").close();
  $("details-content").replaceChildren();
  $("details-title").textContent = "资源详情";
  $("action-dialog").close();
  $("action-form").reset();
  $("action-error").textContent = "";
  $("action-error").hidden = true;
  $("action-title").textContent = "实例操作";
  $("action-description").textContent = "";
  $("action-notice").textContent = "";
  pendingAction = null;
  render();
}

async function refresh(closeConnection = false) {
  if (inFlight) return inFlight;
  const current = generation;
  state.loading = true;
  renderChrome();
  inFlight = (async () => {
    try {
      const [stats, models, workers, blocks, plugins] = await Promise.all([
        api("/admin/stats"),
        api("/admin/models"),
        api("/admin/workers"),
        api("/admin/blacklist"),
        api("/admin/scheduling/plugins"),
      ]);
      if (current !== generation) return false;
      state.data = {
        stats,
        models: models.models || [],
        workers: workers.workers || [],
        blocks: blocks.blacklist || [],
        plugins,
      };
      state.updatedAt = new Date();
      state.stale = false;
      state.error = "";
      state.connected = true;
      if (closeConnection) $("connection-dialog").close();
      return true;
    } catch (err) {
      if (current !== generation) return false;
      if (err.status === 401) {
        clearConnection();
        showConnection(errorMessage(err));
      } else {
        state.stale = true;
        state.error = `${errorMessage(err)}${state.data ? " 当前显示上次快照，管理操作已禁用。" : ""}`;
        if ($("connection-dialog").open) {
          $("connection-error").textContent = errorMessage(err);
          $("connection-error").hidden = false;
        }
      }
      return false;
    } finally {
      if (current === generation) state.loading = false;
      render();
    }
  })();
  try {
    return await inFlight;
  } finally {
    inFlight = null;
  }
}

function canMutate() {
  return Boolean(state.data && !state.stale && !state.busy);
}
function disabled() {
  return canMutate() ? "" : "disabled";
}
function tag(text, kind = "") {
  return `<span class="tag ${e(kind)}">${e(text)}</span>`;
}
function statusTag(worker) {
  const [label, kind] = statuses[workerStatus(worker)];
  return tag(label, kind);
}
function empty(title, description) {
  return `<div class="empty-state panel"><div class="empty-symbol" aria-hidden="true">◫</div><h2>${e(title)}</h2><p>${e(description)}</p></div>`;
}
function rowEmpty(columns, message) {
  return `<tr><td colspan="${columns}" class="empty-row">${e(message)}</td></tr>`;
}
function progress(value, label) {
  if (value === null || !Number.isFinite(value))
    return '<span class="muted">未知容量</span>';
  return `<div class="metric"><progress max="1" value="${Math.max(0, Math.min(value, 1))}" aria-label="${e(label)}"></progress><span>${e(percent(value))}</span></div>`;
}
function detailsButton(kind, id) {
  return `<button class="button small secondary" data-detail="${kind}" data-id="${e(id)}">详情</button>`;
}
function actionButton(action, id, label) {
  const cls =
    action === "block"
      ? "text-danger"
      : action === "unblock" || action === "resume"
        ? "text-green"
        : "secondary";
  return `<button class="button small ${cls}" data-action="${action}" data-id="${e(id)}" ${disabled()}>${e(label)}</button>`;
}

function renderChrome() {
  const d = state.data;
  $("page-title").textContent = pages[state.page][0];
  $("page-description").textContent = pages[state.page][1];
  document.title = `${pages[state.page][0]} · ModelWeave`;
  document.querySelectorAll("[data-page]").forEach((a) => {
    if (a.dataset.page === state.page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("nav-models").textContent = d ? count(d.models.length) : "—";
  $("nav-workers").textContent = d ? count(d.workers.length) : "—";
  $("nav-blocks").textContent = d ? count(d.blocks.length) : "—";
  $("nav-plugins").textContent = d ? count(d.plugins.scorers.length) : "—";
  $("connection-label").textContent = state.stale
    ? "同步失败"
    : d
      ? "已连接"
      : state.loading
        ? "正在连接"
        : "未连接";
  $("connection-dot").className =
    `status-dot${state.stale ? " error-dot" : d ? "" : " muted-dot"}`;
  $("refresh-button").disabled =
    state.loading || state.busy || !state.connected;
  $("content").setAttribute("aria-busy", String(state.loading));
  $("backend-label").textContent = d
    ? `STATE / ${d.stats.state_backend.toUpperCase()}`
    : "等待连接";
  $("version-label").textContent = d ? d.stats.version : "ModelWeave";
  $("updated-at").textContent = state.updatedAt
    ? `${state.stale ? "上次成功同步" : "更新于"} ${state.updatedAt.toLocaleTimeString("zh-CN", { hour12: false })}`
    : "尚未同步";
  $("error-banner").textContent = state.error;
  $("error-banner").hidden = !state.error;
  $("security-banner").hidden = !d || d.stats.admin_auth_required;
}

function render() {
  renderChrome();
  // Replacing a focused input mid-composition would break Chinese/Japanese
  // IMEs, including when a background refresh completes during composition.
  if (composing && state.data) return;
  if (!state.data) composing = false;
  const active = document.activeElement;
  const focus =
    $("content").contains(active) && active.id
      ? {
          id: active.id,
          start: active.selectionStart,
          end: active.selectionEnd,
        }
      : null;
  if (!state.data) {
    $("content").innerHTML = empty(
      state.loading ? "正在读取网关状态" : "连接后查看网关状态",
      state.loading
        ? "正在查询当前网关的管理 API…"
        : "点击右上角「连接设置」，使用管理令牌连接当前网关。",
    );
    return;
  }
  const views = {
    overview: renderOverview,
    models: renderModels,
    workers: renderWorkers,
    blacklist: renderBlacklist,
    tenants: renderTenants,
    plugins: renderPlugins,
  };
  $("content").innerHTML = views[state.page]();
  if (focus && $(focus.id)) {
    $(focus.id).focus({ preventScroll: true });
    if (
      focus.start !== null &&
      typeof $(focus.id).setSelectionRange === "function"
    ) {
      $(focus.id).setSelectionRange(focus.start, focus.end);
    }
  }
}

function renderOverview() {
  const { stats, models, workers, blocks } = state.data;
  const available = workers.filter((w) => workerStatus(w) === "active").length;
  const inflight = workers.reduce((sum, w) => sum + w.inflight, 0);
  const waiting = models.reduce((sum, m) => sum + m.queue_len, 0);
  const unhealthy = workers.filter(
    (w) => workerStatus(w) === "unhealthy",
  ).length;
  const cards = [
    [
      "已配置模型",
      count(models.length),
      "",
      "包含 canonical name 与独立路由配置",
      "▦",
    ],
    [
      "可调度实例",
      count(available),
      `/ ${count(workers.length)}`,
      `${workers.length - available} 个实例已隔离、摘流或处于冷却`,
      "◉",
    ],
    [
      "在途请求",
      count(inflight),
      "",
      `${count(waiting)} 个请求正在等待调度`,
      "↗",
    ],
    [
      "隔离规则",
      count(blocks.length),
      "",
      `${count(unhealthy)} 个实例处于被动健康冷却`,
      "⊘",
    ],
  ];
  const attention = workers.filter((w) => workerStatus(w) !== "active");
  return `<div class="stats-grid">${cards.map(([label, value, sub, caption, mark]) => `<article class="stat-card"><div class="stat-label">${label}<span class="stat-mark" aria-hidden="true">${mark}</span></div><div class="stat-value">${value}<small>${sub}</small></div><div class="stat-caption">${e(caption)}</div></article>`).join("")}</div>
  <div class="overview-grid"><div class="overview-main">
    <section class="panel"><div class="panel-heading"><div><h2>模型运行状态</h2><p>按模型汇总实例池与准入队列</p></div><span class="panel-kicker">MODEL HEALTH</span></div><div class="table-scroll"><table><thead><tr><th>模型 / MODEL</th><th>执行模式</th><th>可调度 / 总数</th><th>队列</th><th>负载</th></tr></thead><tbody>${models.map((m) => `<tr><td><button class="button link model-name" data-model-workers="${e(m.name)}">${e(m.name)}</button><span class="cell-sub">${e(m.aliases.join(" · ") || m.tokenizer_model)}</span></td><td>${tag(m.mode, "blue-tag")}</td><td><span class="count-pair green">${count(m.instances.available)} <span>/ ${count(m.instances.total)}</span></span></td><td class="mono">${count(m.queue_len)} <span class="muted">/ ${count(m.queue_capacity)}</span></td><td>${progress(m.utilization, `${m.name} 负载`)}</td></tr>`).join("")}</tbody></table></div><div class="panel-footer"><span>可调度 = 健康且未摘流、未拉黑；实际准入仍取决于容量。</span><a class="green" href="#models">全部模型 →</a></div></section>
    <section class="panel system-panel"><div class="panel-heading"><div><h2>网关状态</h2><p>当前副本与共享状态后端</p></div>${tag(stats.ready ? "就绪" : "停止接流", stats.ready ? "green-tag" : "amber-tag")}</div><dl class="system-list"><div><dt>网关版本</dt><dd class="mono">${e(stats.version)}</dd></div><div><dt>运行时间</dt><dd>${e(duration(stats.uptime_seconds * 1000))}</dd></div><div><dt>状态 / QoS 后端</dt><dd>${e(stats.state_backend)} / ${e(stats.qos_backend)}</dd></div><div><dt>前缀缓存索引</dt><dd>${e(stats.kv_index.kind)} · ${count(stats.kv_index.blocks)} blocks</dd></div><div><dt>状态共享范围</dt><dd>${stats.state_backend === "redis" ? "同一 Redis namespace 的网关副本" : "当前网关进程（重启后清空）"}</dd></div><div><dt>管理接口认证</dt><dd>${stats.admin_auth_required ? "令牌保护" : "未启用 · 仅限可信网络"}</dd></div></dl></section>
  </div><div class="overview-side"><section class="panel"><div class="panel-heading"><div><h2>调度池分布</h2><p>可调度 / 已注册实例</p></div><span class="panel-kicker">POOLS</span></div><div class="role-pools">${roles
    .map((role) => {
      const ws = workers.filter((w) => w.role === role);
      const n = ws.filter((w) => workerStatus(w) === "active").length;
      return `<div class="role-pool"><div class="role-pool-label"><strong>${role}</strong><span>${n} / ${ws.length}</span></div><progress max="${Math.max(ws.length, 1)}" value="${n}" aria-label="${role} 可调度实例"></progress><div class="role-description">${roleNames[role]}</div></div>`;
    })
    .join(
      "",
    )}</div></section><section class="panel"><div class="panel-heading"><div><h2>需要关注</h2><p>被隔离或暂不可用的实例</p></div>${tag(attention.length)}</div><div class="attention-list">${attention.length ? attention.map((w) => `<div class="attention-item"><button class="button link model-name" data-detail="worker" data-id="${e(w.id)}">${e(w.id)}</button>${statusTag(w)}</div>`).join("") : '<div class="quiet">✓ 当前没有异常或摘流中的实例</div>'}</div></section></div></div>`;
}

function renderModels() {
  const models = filterModels(state.data.models, state.modelQuery);
  return `<div class="filter-bar"><input id="model-search" type="search" placeholder="搜索模型名称、别名或 tokenizer…" aria-label="搜索模型" value="${e(state.modelQuery)}"><span class="section-note">${models.length} 个模型 / ${state.data.models.length} 个已配置</span></div>
  ${models.length ? `<div class="models-grid">${models.map((m) => `<article class="panel model-card"><div class="model-card-head"><h2>${e(m.name)}</h2>${tag(m.mode, "blue-tag")}</div><p class="model-aliases">别名：${e(m.aliases.join(" · ") || "—")}</p><div class="role-tags">${roles.map((role) => `<div class="role-tile"><span>${role}</span><strong>${m.roles[role].available}</strong><small> / ${m.roles[role].total}</small></div>`).join("")}</div><div class="model-card-stats"><span>PD 握手 <strong>${e(m.handshake)}</strong></span><span>上下文 <strong>${m.context_length ? count(m.context_length) : "使用 tokenizer 配置"}</strong></span></div><div class="model-card-stats"><span>排队 <strong>${count(m.queue_len)} / ${count(m.queue_capacity)}</strong></span><span>过载等级 <strong>L${m.overload_level}</strong></span></div><div class="model-card-footer">${detailsButton("model", m.name)}<button class="button link" data-model-workers="${e(m.name)}">查看 ${m.instances.total} 个实例 →</button></div></article>`).join("")}</div>` : empty("没有匹配的模型", "试试模型的 canonical name、别名或 tokenizer 名称。")}`;
}

function selectOptions(items, selected) {
  return items
    .map(
      ([value, label]) =>
        `<option value="${e(value)}" ${selected === value ? "selected" : ""}>${e(label)}</option>`,
    )
    .join("");
}

function renderWorkers() {
  const workers = filterWorkers(state.data.workers, state.filters);
  return `<section class="panel"><div class="toolbar"><input id="worker-search" class="search-input" type="search" aria-label="搜索实例" placeholder="搜索实例 ID、地址、标签…" value="${e(state.filters.query)}"><select id="worker-model" aria-label="按模型筛选">${selectOptions([["", "全部模型"], ...state.data.models.map((m) => [m.name, m.name])], state.filters.model)}</select><select id="worker-role" aria-label="按角色筛选">${selectOptions([["", "全部角色"], ...roles.map((r) => [r, r])], state.filters.role)}</select><select id="worker-status" aria-label="按状态筛选">${selectOptions([["", "全部状态"], ...Object.entries(statuses).map(([k, v]) => [k, v[0]])], state.filters.status)}</select><span class="toolbar-count">${workers.length} 个实例</span></div>
  <div class="table-scroll"><table><thead><tr><th>实例 / 模型</th><th>角色</th><th>状态</th><th>在途 / 上限</th><th>KV 占用</th><th>状态报告</th><th>操作</th></tr></thead><tbody>${workers.length ? workers.map((w) => `<tr data-worker-row="${e(w.id)}"><td><span class="model-name mono">${e(w.id)}</span><span class="cell-sub" title="${e(w.target)}">${e(w.model)} · ${e(w.target)}</span></td><td>${tag(w.role)}</td><td>${statusTag(w)}</td><td class="mono">${count(w.inflight)} <span class="muted">/ ${w.max_running_requests ? count(w.max_running_requests) : "—"}</span></td><td>${progress(kvRatio(w), `${w.id} KV 占用`)}</td><td>${w.state_status === "fresh" ? `<span class="green">${e(duration(w.state_age_ms || 0))}前</span>` : tag(w.state_status === "stale" ? "报告已过期" : "无有效报告", "amber-tag")}<span class="cell-sub">${w.source === "heartbeat" ? `心跳 ${e(duration(w.heartbeat_age_ms))}前` : "静态注册"}</span></td><td><div class="row-actions">${detailsButton("worker", w.id)}${actionButton(w.admin_drained ? "resume" : "drain", w.id, w.admin_drained ? "恢复" : "摘流")}${actionButton(w.blocked ? "unblock" : "block", w.id, w.blocked ? "解除拉黑" : "拉黑")}</div></td></tr>`).join("") : rowEmpty(7, state.data.workers.length ? "没有符合筛选条件的实例" : "尚无注册实例；请接入 mwip 引擎或配置静态实例")}</tbody></table></div><div class="table-note">摘流仅作用于当前注册；拉黑按实例 ID 持续生效。两者都不会中断已在执行的请求。完成与错误计数仅统计当前网关副本。</div></section>`;
}

function renderBlacklist() {
  const q = state.blockQuery.trim().toLowerCase();
  const blocks = state.data.blocks.filter((b) =>
    [b.id, b.reason].some((v) => v.toLowerCase().includes(q)),
  );
  const online = new Map(state.data.workers.map((w) => [w.id, w]));
  return `<div class="filter-bar"><input id="block-search" type="search" placeholder="搜索实例 ID 或拉黑原因…" aria-label="搜索黑名单" value="${e(state.blockQuery)}"><button class="button primary" data-action="block" data-id="" ${disabled()}>＋ 添加隔离规则</button></div><section class="panel"><div class="table-scroll"><table><thead><tr><th>实例 ID</th><th>注册状态</th><th>拉黑原因</th><th>规则更新时间</th><th>操作</th></tr></thead><tbody>${blocks.length ? blocks.map((b) => `<tr data-block-row="${e(b.id)}"><td class="model-name mono">${e(b.id)}</td><td>${online.has(b.id) ? tag("已注册 · 已隔离", "red-tag") : tag("离线 / 未注册")}<span class="cell-sub">${e(online.get(b.id)?.model || "重新注册后仍保持隔离")}</span></td><td class="reason-cell">${e(b.reason)}</td><td class="mono">${e(new Date(b.blocked_at_ms).toLocaleString("zh-CN", { hour12: false }))}</td><td>${actionButton("unblock", b.id, "解除拉黑")}</td></tr>`).join("") : rowEmpty(5, state.data.blocks.length ? "没有匹配的隔离规则" : "黑名单为空，暂无被主动隔离的实例")}</tbody></table></div><div class="table-note">${state.data.stats.state_backend === "redis" ? "规则在同一 Redis namespace 的所有网关副本生效；持久性取决于 Redis 的持久化/高可用配置。" : "当前为 memory 模式：规则仅在此进程生效，网关重启后清空。多副本或持久隔离请使用 Redis。"} 可在实例离线或注册前添加规则。</div></section>`;
}

function renderTenants() {
  const tenants = [...(state.data.stats.tenants || [])].sort((a, b) =>
    a.name.localeCompare(b.name),
  );
  const balance = (limit, available) =>
    limit
      ? `${count(Math.floor(available || 0))} <span class="muted">/ ${count(limit)}</span>`
      : '<span class="muted">不限额</span>';
  return `<p class="section-note">${state.data.stats.qos_backend === "redis" ? "以下并发与余额在整个 Redis namespace 共享，不按网关副本数翻倍。" : "以下限额与余额仅作用于当前网关进程。"} API 密钥不会返回到管理页面。</p><section class="panel"><div class="table-scroll"><table><thead><tr><th>租户</th><th>优先级</th><th>并发占用 / 上限</th><th>RPM 可用 / 限额</th><th>TPM 可用 / 限额</th></tr></thead><tbody>${tenants.length ? tenants.map((t) => `<tr><td class="model-name">${e(t.name)}</td><td>${tag(t.priority, "blue-tag")}</td><td class="mono">${count(t.inflight)} <span class="muted">/ ${t.max_concurrency ? count(t.max_concurrency) : "不限额"}</span></td><td class="mono">${balance(t.rpm_limit, t.rpm_available)}</td><td class="mono">${balance(t.tpm_limit, t.tpm_available)}</td></tr>`).join("") : rowEmpty(5, "未配置租户配额")}</tbody></table></div><div class="table-note">配额配置是只读的；修改租户、密钥、模型和调度策略仍通过网关 YAML 配置发布。</div></section>`;
}

function composition(policy) {
  const scorers = policy.scorers || [];
  return `<div class="composition-name mono">${e(policy.name)}</div><div class="composition-terms">${scorers.length ? scorers.map((s) => `<span class="composition-chip${s.weight === 0 ? " disabled-term" : ""}" title="${e(s.params ? JSON.stringify(s.params) : "无参数")}">${e(s.name)} <strong>× ${e(s.weight)}</strong></span>`).join("") : '<span class="muted">独立选路策略</span>'}</div>`;
}

function renderPlugins() {
  const { plugins, models } = state.data;
  const matches = filterPlugins(plugins.scorers, state.pluginQuery);
  return `<div class="plugin-notice"><strong>编译期插件 · 按模型组合</strong><span>总分 = Σ（算法权重 × 评分），权重是相对系数而非概率。配置由网关 YAML 发布并在重启后生效；本页不直接修改线上调度。</span></div>
  <div class="filter-bar"><input id="plugin-search" type="search" aria-label="搜索调度插件" placeholder="搜索算法名称、目录或参数…" value="${e(state.pluginQuery)}"><span class="section-note">${matches.length} / ${plugins.scorers.length} 个评分插件</span></div>
  <div class="models-grid plugins-grid">${matches
    .map((plugin) => {
      const usages = models.flatMap((m) =>
        roles
          .filter((role) =>
            (m.policies[role].scorers || []).some(
              (s) => s.name === plugin.name && s.weight > 0,
            ),
          )
          .map((role) => [m.name, role]),
      );
      return `<article class="panel plugin-card" data-plugin="${e(plugin.name)}"><div class="model-card-head"><h2>${e(plugin.name)}</h2>${tag(plugin.version, "blue-tag")}</div><p class="plugin-description">${e(plugin.description)}</p><div class="plugin-source mono">${e(plugin.source || "compiled-in extension")}</div><div class="plugin-parameters">${plugin.parameters.length ? plugin.parameters.map((p) => `<span class="tag" title="${e(p.description)}">${e(p.name)} · ${e(p.type)}${p.required ? " *" : ""}</span>`).join("") : '<span class="muted">无配置参数</span>'}</div><div class="model-card-footer"><span class="plugin-usage">${new Set(usages.map(([name]) => name)).size} 个模型 / ${usages.length} 个角色启用</span><button class="button link" data-plugin-example="${e(plugin.name)}">配置示例 →</button></div></article>`;
    })
    .join(
      "",
    )}</div>${matches.length ? "" : empty("没有匹配的调度插件", "新插件需在仓库内注册并重新构建网关，不能仅上传代码文件。")}
  <section class="panel composition-panel"><div class="panel-heading"><div><h2>各模型生效组合</h2><p>包含模型默认值与角色覆盖后的实际算法、权重；灰色项已通过零权重禁用。</p></div><span class="panel-kicker">EFFECTIVE ROUTING</span></div><div class="table-scroll"><table><thead><tr><th>模型</th>${roles.map((r) => `<th>${r}</th>`).join("")}</tr></thead><tbody>${models.map((m) => `<tr data-composition="${e(m.name)}"><td><button class="button link model-name" data-detail="model" data-id="${e(m.name)}">${e(m.name)}</button></td>${roles.map((r) => `<td>${composition(m.policies[r])}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>
  <section class="panel policy-panel"><div class="panel-heading"><div><h2>可用选路策略</h2><p>评分插件通过 weighted 组合；轮询、随机等独立策略不能当作评分项混合。</p></div></div><div class="policy-chips">${plugins.policies.map((name) => tag(name)).join("")}</div></section>`;
}

function showPluginExample(name) {
  const plugin = state.data.plugins.scorers.find((p) => p.name === name);
  if (!plugin) return;
  $("details-title").textContent = `${name} · 配置示例`;
  $("details-content").innerHTML =
    `<p class="dialog-description">将模型名和参数替换为实际值，合并至网关 YAML，再校验并发布新配置。角色配置会完整覆盖 routing.default，不合并算法列表。</p><pre>${e(pluginExample(plugin))}</pre><h3 class="detail-section">插件参数</h3><pre>${e(JSON.stringify(plugin.parameters, null, 2))}</pre>`;
  $("details-dialog").showModal();
}

function showDetails(kind, id) {
  const item =
    kind === "model"
      ? state.data.models.find((m) => m.name === id)
      : state.data.workers.find((w) => w.id === id);
  if (!item) return;
  $("details-title").textContent = id;
  const entries =
    kind === "model"
      ? [
          ["模型别名", item.aliases.join(" · ") || "—"],
          ["执行模式 / 握手", `${item.mode} / ${item.handshake}`],
          ["Tokenizer 模型", item.tokenizer_model],
          ["引擎模型", item.engine_model || "未指定"],
          ["上下文上限", item.context_length || "使用 tokenizer 配置"],
          ["最大队列等待", duration(item.max_wait_ms)],
        ]
      : [
          ["模型 / 角色", `${item.model} / ${item.role}`],
          ["mwip 目标", item.target],
          ["注册来源", item.source],
          ["当前状态", statuses[workerStatus(item)][0]],
          ["管理摘流", item.admin_drained ? "是" : "否"],
          ["状态报告", item.state_status],
          ["完成请求（当前副本）", count(item.completed)],
          ["传输错误（当前副本）", count(item.errors)],
          [
            "KV 已用 / 容量",
            `${count(item.effective_kv_used_tokens)} / ${count(item.kv_capacity_tokens)}`,
          ],
          ["拉黑原因", item.block?.reason || "—"],
        ];
  $("details-content").innerHTML =
    `<dl class="detail-grid">${entries.map(([k, v]) => `<div><dt>${e(k)}</dt><dd>${e(v)}</dd></div>`).join("")}</dl>${kind === "model" ? `<h3 class="detail-section">各角色调度策略与权重</h3><pre>${e(JSON.stringify(item.policies, null, 2))}</pre>` : `<h3 class="detail-section">实例标签</h3><pre>${e(JSON.stringify(item.labels || {}, null, 2))}</pre>`}<h3 class="detail-section">完整 API 快照（查询时）</h3><pre>${e(JSON.stringify(item, null, 2))}</pre>`;
  $("details-dialog").showModal();
}

function showAction(action, id) {
  if (!canMutate()) return;
  pendingAction = { action, id };
  $("action-fields").disabled = false;
  $("action-title").textContent = actions[action][0];
  $("action-description").textContent = actions[action][2];
  $("confirm-action").textContent = actions[action][1];
  $("confirm-action").className =
    `button ${action === "block" || action === "drain" ? "danger" : "primary"}`;
  $("action-worker").value = id;
  $("action-worker").readOnly = Boolean(id);
  $("reason-field").hidden = action !== "block";
  $("action-reason").required = action === "block";
  $("action-reason").value = "";
  $("action-error").hidden = true;
  $("action-notice").textContent =
    `${state.data.stats.state_backend === "redis" ? "影响同一 Redis namespace 的所有网关副本。" : "仅影响当前网关进程，重启后清空。"} ${action === "unblock" || action === "resume" ? "这不会清除其他隔离、引擎摘流或健康冷却状态；恢复后仍需满足准入条件。" : action === "drain" ? "当前注册被删除或过期后，摘流标记会消失。长期隔离请使用拉黑。" : "规则持续生效，必须由管理员显式解除。"}`;
  $("action-dialog").showModal();
  if (!id) $("action-worker").focus();
  else if (action === "block") $("action-reason").focus();
  else $("cancel-action").focus();
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => {
    $("toast").hidden = true;
  }, 4500);
}

$("connection-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = event.submitter;
  submit.disabled = true;
  $("admin-token").disabled = true;
  const nextToken = $("admin-token").value.trim();
  $("admin-token").value = "";
  // Ignore any prior connection's late responses before installing a new token.
  generation++;
  if (inFlight) await inFlight;
  token = nextToken;
  state.connected = true;
  try {
    await refresh(true);
  } finally {
    submit.disabled = false;
    $("admin-token").disabled = false;
    if ($("connection-dialog").open) $("admin-token").focus();
  }
});
$("connection-button").addEventListener("click", () => showConnection());
$("disconnect-button").addEventListener("click", () => {
  clearConnection();
  $("connection-dialog").close();
});
$("refresh-button").addEventListener("click", () => {
  void refresh();
});
$("close-details").addEventListener("click", () => $("details-dialog").close());
$("cancel-action").addEventListener("click", () => $("action-dialog").close());
$("action-dialog").addEventListener("cancel", (event) => {
  if (state.busy) event.preventDefault();
});
$("action-dialog").addEventListener("close", () => {
  if (!state.busy) pendingAction = null;
});

$("action-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!pendingAction || !canMutate()) return;
  let spec;
  try {
    spec = mutationSpec(
      pendingAction.action,
      $("action-worker").value,
      $("action-reason").value,
    );
  } catch (err) {
    $("action-error").textContent = errorMessage(err);
    $("action-error").hidden = false;
    return;
  }
  const current = generation;
  state.busy = true;
  $("action-fields").disabled = true;
  render();
  try {
    // A pre-mutation refresh must not overwrite the post-mutation result.
    if (inFlight) await inFlight;
    if (current !== generation) return;
    await api(spec.path, spec);
    if (current !== generation) return;
    $("action-dialog").close();
    pendingAction = null;
    toast("操作已生效，正在同步最新状态。");
    await refresh();
  } catch (err) {
    if (current !== generation) return;
    if (err.status === 401) {
      clearConnection();
      showConnection(errorMessage(err));
    } else {
      $("action-error").textContent =
        `${errorMessage(err)} 操作未确认成功，请刷新状态后再试。`;
      $("action-error").hidden = false;
      if (!err.status || err.status >= 500) {
        state.stale = true;
        state.error = "操作结果未确认；请刷新网关状态后再执行管理操作。";
      }
    }
  } finally {
    state.busy = false;
    $("action-fields").disabled = false;
    render();
  }
});

$("content").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.disabled || !state.data) return;
  if (button.dataset.detail)
    showDetails(button.dataset.detail, button.dataset.id);
  else if (button.dataset.pluginExample)
    showPluginExample(button.dataset.pluginExample);
  else if (button.dataset.action)
    showAction(button.dataset.action, button.dataset.id);
  else if (button.dataset.modelWorkers !== undefined) {
    state.filters = {
      query: "",
      role: "",
      status: "",
      model: button.dataset.modelWorkers,
    };
    if (state.page === "workers") render();
    else location.hash = "workers";
  }
});
function updateSearch(event) {
  if (composing || event.isComposing) return;
  const { id, value } = event.target;
  if (id === "model-search") state.modelQuery = value;
  else if (id === "block-search") state.blockQuery = value;
  else if (id === "plugin-search") state.pluginQuery = value;
  else if (id === "worker-search") state.filters.query = value;
  else return;
  render();
}
$("content").addEventListener("input", updateSearch);
$("content").addEventListener("compositionstart", () => {
  composing = true;
});
$("content").addEventListener("compositionend", (event) => {
  composing = false;
  updateSearch(event);
});
$("content").addEventListener("change", (event) => {
  const key = {
    "worker-model": "model",
    "worker-role": "role",
    "worker-status": "status",
  }[event.target.id];
  if (key) {
    state.filters[key] = event.target.value;
    render();
  }
});
function navigate() {
  const page = location.hash.slice(1);
  state.page = Object.hasOwn(pages, page) ? page : "overview";
  render();
}
window.addEventListener("hashchange", navigate);
document.addEventListener("visibilitychange", () => {
  if (
    !document.hidden &&
    state.connected &&
    !state.busy &&
    !$("connection-dialog").open &&
    $("auto-refresh").checked
  )
    void refresh();
});
setInterval(() => {
  if (
    !document.hidden &&
    state.connected &&
    !state.busy &&
    !$("connection-dialog").open &&
    $("auto-refresh").checked
  )
    void refresh();
}, 5000);
navigate();
void refresh();
