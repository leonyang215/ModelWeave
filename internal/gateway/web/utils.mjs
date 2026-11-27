// Shared, dependency-free view logic. Keep untrusted model/instance metadata
// out of HTML, URLs and CSS unless it has been encoded for that context.
export function escapeHTML(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c],
  );
}

export function count(value) {
  return Number.isFinite(Number(value))
    ? Number(value).toLocaleString("en-US")
    : "—";
}

export function percent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

export function duration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} 秒`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)} 分钟`;
  if (ms < 86400000) return `${Math.floor(ms / 3600000)} 小时`;
  return `${Math.floor(ms / 86400000)} 天`;
}

export function workerStatus(w) {
  if (w.blocked) return "blocked";
  if (w.draining) return "draining";
  if (!w.healthy) return "unhealthy";
  return "active";
}

export function filterWorkers(workers, filters) {
  const query = (filters.query || "").trim().toLowerCase();
  return workers.filter(
    (w) =>
      (!filters.model || w.model === filters.model) &&
      (!filters.role || w.role === filters.role) &&
      (!filters.status || workerStatus(w) === filters.status) &&
      (!query ||
        [w.id, w.model, w.target, ...Object.values(w.labels || {})].some((v) =>
          String(v).toLowerCase().includes(query),
        )),
  );
}

export function filterModels(models, query) {
  const q = query.trim().toLowerCase();
  return models.filter((m) =>
    [m.name, ...(m.aliases || []), m.tokenizer_model].some((v) =>
      String(v || "")
        .toLowerCase()
        .includes(q),
    ),
  );
}

export function kvRatio(w) {
  return w.kv_capacity_tokens > 0
    ? w.effective_kv_used_tokens / w.kv_capacity_tokens
    : null;
}

export function filterPlugins(plugins, query) {
  const q = query.trim().toLowerCase();
  return plugins.filter((plugin) =>
    [
      plugin.name,
      plugin.description,
      plugin.source,
      ...(plugin.parameters || []).map((p) => p.name),
    ].some((v) =>
      String(v || "")
        .toLowerCase()
        .includes(q),
    ),
  );
}

// Generates an editable YAML template, not a live routing mutation. JSON
// quoted strings are valid YAML scalars, including unusual plugin names.
export function pluginExample(plugin) {
  const params = plugin.parameters || [];
  const lines = [
    "models:",
    '  - name: "your-model"',
    "    routing:",
    "      default:",
    "        policy: weighted",
    "        scorers:",
    `          - name: ${JSON.stringify(plugin.name)}`,
    `            weight: ${plugin.name === "load" ? 1 : 2}`,
  ];
  if (params.length) {
    lines.push("            params:");
    for (const param of params) {
      const value =
        plugin.name === "label" && param.name === "key"
          ? "zone"
          : plugin.name === "label" && param.name === "value"
            ? "west"
            : `<${param.name}>`;
      lines.push(
        `              ${JSON.stringify(param.name)}: ${JSON.stringify(value)}`,
      );
    }
  }
  if (plugin.name !== "load")
    lines.push('          - name: "load"', "            weight: 1");
  lines.push('      encode: {policy: "load"}');
  return lines.join("\n");
}

export function mutationSpec(action, id, reason = "") {
  if (!id.trim() || new TextEncoder().encode(id).length > 256) {
    throw new Error("实例 ID 不能为空，且不能超过 256 字节。");
  }
  const key = encodeURIComponent(id);
  switch (action) {
    case "block": {
      const text = reason.trim();
      if (!text || Array.from(text).length > 512)
        throw new Error("请填写 1 至 512 个字符的拉黑原因。");
      return {
        method: "PUT",
        path: `/admin/blacklist/${key}`,
        body: { reason: text },
      };
    }
    case "unblock":
      return { method: "DELETE", path: `/admin/blacklist/${key}` };
    case "drain":
    case "resume":
      return { method: "POST", path: `/admin/workers/${key}/${action}` };
    default:
      throw new Error("未知实例操作。");
  }
}
