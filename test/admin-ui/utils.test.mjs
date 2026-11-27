import test from "node:test";
import assert from "node:assert/strict";
import {
  escapeHTML,
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
} from "../../internal/gateway/web/utils.mjs";

test("all untrusted HTML characters are escaped, including attribute quotes", () => {
  assert.equal(
    escapeHTML(`<img src=x onerror="alert('x')">&`),
    "&lt;img src=x onerror=&quot;alert(&#39;x&#39;)&quot;&gt;&amp;",
  );
  assert.equal(escapeHTML(null), "");
  assert.equal(escapeHTML("模型"), "模型");
});

test("status precedence matches the admin API without overwriting independent flags", () => {
  assert.equal(
    workerStatus({ blocked: true, draining: true, healthy: false }),
    "blocked",
  );
  assert.equal(workerStatus({ draining: true, healthy: false }), "draining");
  assert.equal(workerStatus({ healthy: false }), "unhealthy");
  assert.equal(workerStatus({ healthy: true }), "active");
});

test("instance filters compose across models, roles, status, addresses and labels", () => {
  const workers = [
    {
      id: "d0",
      model: "m",
      role: "decode",
      target: "HOST:9001",
      healthy: true,
      labels: { zone: "WEST" },
    },
    {
      id: "d1",
      model: "m",
      role: "decode",
      target: "other:9001",
      healthy: true,
      blocked: true,
    },
    {
      id: "p0",
      model: "other",
      role: "prefill",
      target: "third:9001",
      healthy: true,
    },
  ];
  assert.equal(filterWorkers(workers, {}).length, 3);
  assert.deepEqual(
    filterWorkers(workers, {
      model: "m",
      role: "decode",
      status: "active",
      query: " west ",
    }).map((w) => w.id),
    ["d0"],
  );
  assert.deepEqual(
    filterWorkers(workers, { status: "blocked" }).map((w) => w.id),
    ["d1"],
  );
  assert.equal(filterWorkers(workers, { query: "host" }).length, 1);
  assert.equal(filterWorkers(workers, { model: "missing" }).length, 0);
});

test("model search resolves aliases and tokenizer names", () => {
  const models = [
    { name: "m", aliases: ["DeepSeek-Flash"], tokenizer_model: "tokenizer-m" },
  ];
  for (const query of ["deepseek", " M ", "tokenizer-m", ""])
    assert.equal(filterModels(models, query).length, 1);
  assert.equal(filterModels(models, "unknown").length, 0);
});

test("unknown KV capacity is not displayed as zero utilization", () => {
  assert.equal(
    kvRatio({ kv_capacity_tokens: 0, effective_kv_used_tokens: 50 }),
    null,
  );
  assert.equal(
    kvRatio({ kv_capacity_tokens: 100, effective_kv_used_tokens: 125 }),
    1.25,
  );
  assert.equal(percent(1.25), "125.0%");
  assert.equal(percent(NaN), "—");
});

test("formatters handle stale, zero and invalid values", () => {
  assert.equal(count(12345), "12,345");
  assert.equal(count(Infinity), "—");
  assert.equal(duration(0), "0 ms");
  assert.equal(duration(1200), "1.2 秒");
  assert.equal(duration(120000), "2 分钟");
  assert.equal(duration(7200000), "2 小时");
  assert.equal(duration(172800000), "2 天");
  assert.equal(duration(undefined), "—");
});

test("operation URLs encode the entire instance ID, never interpreting it as a path", () => {
  const id = "worker/space ?#中文";
  const spec = mutationSpec("block", id, "  故障排查  ");
  assert.equal(spec.method, "PUT");
  assert.equal(spec.path, `/admin/blacklist/${encodeURIComponent(id)}`);
  assert.deepEqual(spec.body, { reason: "故障排查" });
  assert.equal(mutationSpec("unblock", id).method, "DELETE");
  assert.equal(
    mutationSpec("resume", id).path,
    `/admin/workers/${encodeURIComponent(id)}/resume`,
  );
  assert.equal(mutationSpec("drain", id).method, "POST");
});

test("invalid operations, long IDs and blank reasons cannot be submitted", () => {
  assert.throws(() => mutationSpec("delete", "w0"), /未知/);
  assert.throws(() => mutationSpec("block", "w0", "   "), /原因/);
  assert.throws(() => mutationSpec("block", "w0", "字".repeat(513)), /原因/);
  assert.throws(() => mutationSpec("block", "字".repeat(100), "reason"), /256/);
  assert.throws(() => mutationSpec("block", " ", "reason"), /不能为空/);
  assert.equal(
    mutationSpec("block", "w0", "字".repeat(512)).body.reason.length,
    512,
  );
});

test("plugin catalog search includes directory and documented parameter names", () => {
  const plugins = [
    {
      name: "label",
      description: "Soft preference",
      source: "internal/scheduler/plugins/label",
      parameters: [{ name: "key" }, { name: "value" }],
    },
    {
      name: "load",
      description: "Capacity",
      source: "internal/scheduler/plugins/load",
      parameters: [],
    },
  ];
  for (const query of [" LABEL ", "value", "scheduler/plugins/label", "soft"]) {
    assert.deepEqual(
      filterPlugins(plugins, query).map((p) => p.name),
      ["label"],
    );
  }
  assert.equal(filterPlugins(plugins, "").length, 2);
  assert.equal(filterPlugins(plugins, "unknown").length, 0);
});

test("plugin YAML templates use explicit weights and safe quoted identifiers", () => {
  const label = pluginExample({
    name: "label",
    parameters: [{ name: "key" }, { name: "value" }],
  });
  assert.match(label, /routing:\n      default:\n        policy: weighted/);
  assert.match(label, /weight: 2/);
  assert.match(label, /"key": "zone"/);
  assert.match(label, /"value": "west"/);
  assert.equal(
    (pluginExample({ name: "load" }).match(/name: "load"/g) || []).length,
    1,
  );
  const strange = 'custom"\n<script>';
  assert.ok(pluginExample({ name: strange }).includes(JSON.stringify(strange)));
});
