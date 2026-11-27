import test from "node:test";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const base = process.env.MW_ADMIN_URL;
const token = process.env.MW_ADMIN_TOKEN;
const artifacts =
  process.env.MW_ADMIN_ARTIFACTS ||
  fileURLToPath(new URL("../../tmp/admin-ui/", import.meta.url));

test(
  "embedded admin console: auth, real mutations, safety, and responsive views",
  {
    skip: !base,
    timeout: 120000,
  },
  async (t) => {
    const browser = await chromium.launch({
      args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1024 },
    });
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    const errors = [];
    const cspErrors = [];
    const external = [];
    page.on("pageerror", (err) => {
      errors.push(err.message);
      console.error("Browser JavaScript error:", err.message);
    });
    page.on("console", (msg) => {
      if (/content security policy|violates.*directive/i.test(msg.text()))
        cspErrors.push(msg.text());
    });
    page.on("request", (request) => {
      if (!request.url().startsWith(base)) external.push(request.url());
    });
    const visible = (selector) =>
      page.locator(selector).waitFor({ state: "visible" });
    const connect = async () => {
      await visible("#connection-dialog[open]");
      await page.locator("#admin-token").fill(token);
      await page.locator("#connection-form button[type=submit]").click();
      await page.locator("#connection-dialog").waitFor({ state: "hidden" });
      assert.equal(await page.locator("#admin-token").inputValue(), "");
    };
    const worker = async (id) => {
      const response = await page.request.get(
        `${base}/admin/workers/${encodeURIComponent(id)}`,
        { headers: { "X-ModelWeave-Token": token } },
      );
      assert.equal(response.status(), 200);
      return response.json();
    };
    const row = () => page.locator('[data-worker-row="decode-0"]');
    const submitAction = async (action) => {
      const response = page.waitForResponse(
        (r) => r.request().method() !== "GET" && r.url().includes(action),
      );
      await page.locator("#confirm-action").click();
      assert.equal((await response).status(), 200);
      await page.locator("#action-dialog").waitFor({ state: "hidden" });
      await page.waitForFunction(
        () =>
          document.querySelector("#content").getAttribute("aria-busy") ===
          "false",
      );
    };

    try {
      await t.test(
        "authentication does not expose data or persist credentials",
        async () => {
          await page.goto(`${base}/admin/`);
          await visible("#connection-dialog[open]");
          assert.equal(
            await page
              .locator("#content")
              .innerText()
              .then((s) => s.includes("decode-0")),
            false,
          );
          await page.locator("#admin-token").fill("incorrect-token");
          const denied = page.waitForResponse(
            (r) =>
              r.url() === `${base}/admin/stats` &&
              r.request().headers()["x-modelweave-token"] === "incorrect-token",
          );
          await page.locator("#connection-form button[type=submit]").click();
          assert.equal((await denied).status(), 401);
          await page.waitForFunction(
            () =>
              !document.querySelector("#connection-error").hidden &&
              !document.querySelector("#admin-token").disabled,
          );
          assert.match(
            await page.locator("#connection-error").innerText(),
            /令牌/,
          );
          await connect();
          assert.equal(
            await page.locator("#page-title").innerText(),
            "运行概览",
          );
          assert.equal(
            await page.evaluate(
              () => localStorage.length + sessionStorage.length,
            ),
            0,
          );
          assert.deepEqual(await context.cookies(), []);
          assert.equal((await page.content()).includes(token), false);
          assert.equal(page.url().includes(token), false);
          await page.locator("#auto-refresh").uncheck();
          await mkdir(artifacts, { recursive: true });
          await page.screenshot({
            path: `${artifacts}/overview-desktop.png`,
            fullPage: true,
          });
        },
      );

      await t.test(
        "model aliases, instance filtering, and escaped details",
        async () => {
          await page.locator('a[data-page="models"]').click();
          await visible("#model-search");
          await page
            .locator("#model-search")
            .pressSequentially("DeepSeek-V4.1-Flash");
          assert.equal(await page.locator(".model-card").count(), 1);
          await page.locator("#model-search").evaluate((input) => {
            window.__compositionInput = input;
            input.dispatchEvent(
              new CompositionEvent("compositionstart", { bubbles: true }),
            );
            input.value = "不存在的模型";
            input.dispatchEvent(
              new InputEvent("input", { bubbles: true, isComposing: true }),
            );
          });
          assert.equal(
            await page
              .locator("#model-search")
              .evaluate((input) => input === window.__compositionInput),
            true,
          );
          await page.locator("#model-search").dispatchEvent("compositionend");
          assert.equal(await page.locator(".model-card").count(), 0);
          await page.locator("#model-search").fill("DeepSeek-V4.1-Flash");
          assert.equal(await page.locator(".model-card").count(), 1);
          await page.locator("[data-model-workers]").click();
          await visible("#worker-model");
          assert.equal(
            await page.locator("#worker-model").inputValue(),
            "deepseek-v4.1",
          );
          assert.equal(await page.locator("[data-worker-row]").count(), 4);
          await page.locator("#worker-role").selectOption("decode");
          assert.equal(await page.locator("[data-worker-row]").count(), 2);
          await page.locator("#worker-status").selectOption("active");
          assert.equal(await page.locator("[data-worker-row]").count(), 1);
          await row().locator('[data-detail="worker"]').click();
          await visible("#details-dialog[open]");
          assert.match(
            await page.locator("#details-content").innerText(),
            /<img src=x/,
          );
          assert.equal(await page.locator("#details-content img").count(), 0);
          assert.equal(await page.evaluate(() => window.__mwXSS), undefined);
          await page.locator("#close-details").click();
          await page.locator("#worker-status").selectOption("");
          await page.locator("#worker-role").selectOption("");
          await page.locator("#worker-model").selectOption("");
          await page.screenshot({
            path: `${artifacts}/instances-desktop.png`,
            fullPage: true,
          });
        },
      );

      await t.test(
        "confirmed drain/block/resume/unblock call real gateway APIs",
        async () => {
          await row().locator('[data-action="drain"]').click();
          await page.locator("#cancel-action").click();
          assert.equal(
            Boolean((await worker("decode-0")).admin_drained),
            false,
          );
          await row().locator('[data-action="drain"]').click();
          await submitAction("/drain");
          assert.equal((await worker("decode-0")).admin_drained, true);
          await row().locator('[data-action="block"]').click();
          await page
            .locator("#action-reason")
            .fill("浏览器测试：隔离异常推理实例");
          await submitAction("/admin/blacklist/");
          assert.equal((await worker("decode-0")).blocked, true);
          await row().locator('[data-action="resume"]').click();
          await submitAction("/resume");
          const isolated = await worker("decode-0");
          assert.equal(isolated.blocked, true);
          assert.equal(Boolean(isolated.admin_drained), false);
          await row().locator('[data-action="unblock"]').click();
          await submitAction("/admin/blacklist/");
          assert.equal((await worker("decode-0")).blocked, false);
          await row().locator('[data-action="drain"]').waitFor();
        },
      );

      await t.test(
        "offline rules support encoded IDs and survive page reconnect",
        async () => {
          await page.locator('a[data-page="blacklist"]').click();
          await visible("#block-search");
          await page.locator('#content [data-action="block"]').click();
          await page.locator("#action-worker").fill("offline/new ?#");
          await page
            .locator("#action-reason")
            .fill("<script>window.__mwXSS=true</script>");
          await submitAction("/admin/blacklist/");
          const offline = page.locator('[data-block-row="offline/new ?#"]');
          await offline.waitFor();
          assert.match(await offline.innerText(), /离线/);
          assert.equal(await offline.locator("script").count(), 0);
          await page.reload();
          await connect();
          await page.locator("#auto-refresh").uncheck();
          await offline.waitFor();
          await offline.locator('[data-action="unblock"]').click();
          await submitAction("/admin/blacklist/");
          await offline.waitFor({ state: "detached" });
        },
      );

      await t.test(
        "outages disable mutations, retain a marked snapshot, and recover",
        async () => {
          const fail = (route) =>
            route.fulfill({
              status: 503,
              contentType: "application/json",
              body: '{"message":"state unavailable"}',
            });
          await page.route("**/admin/stats", fail);
          await page.locator("#refresh-button").click();
          await visible("#error-banner");
          assert.match(await page.locator("#error-banner").innerText(), /快照/);
          assert.equal(
            await page.locator('#content [data-action="block"]').isDisabled(),
            true,
          );
          await page.unroute("**/admin/stats", fail);
          await page.locator("#refresh-button").click();
          await page.locator("#error-banner").waitFor({ state: "hidden" });
          assert.equal(
            await page.locator('#content [data-action="block"]').isEnabled(),
            true,
          );
          await page.locator('a[data-page="tenants"]').click();
          await page
            .getByRole("cell", { name: "inference-team", exact: true })
            .waitFor();
          assert.match(
            await page.locator("#content").innerText(),
            /inference-team/,
          );
          assert.equal(
            (await page.locator("#content").innerText()).includes(
              "test-not-a-real-api-key",
            ),
            false,
          );
        },
      );

      await t.test(
        "mobile layout and credential expiry leave no stale privileged content",
        async () => {
          await page.locator('a[data-page="overview"]').click();
          await page.setViewportSize({ width: 390, height: 844 });
          await visible(".stats-grid");
          assert.equal(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= window.innerWidth,
            ),
            true,
          );
          await page.locator("#toast").waitFor({ state: "hidden" });
          await page.screenshot({
            path: `${artifacts}/overview-mobile.png`,
            fullPage: true,
          });
          await page.locator('a[data-page="workers"]').click();
          await visible("#worker-search");
          assert.equal(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= window.innerWidth,
            ),
            true,
          );
          const expire = (route) =>
            route.fulfill({
              status: 401,
              contentType: "application/json",
              body: '{"message":"unauthorized"}',
            });
          await page.route("**/admin/stats", expire);
          await page
            .locator("#worker-search")
            .dispatchEvent("compositionstart");
          await page.locator("#refresh-button").click();
          await visible("#connection-dialog[open]");
          assert.equal(await page.locator("[data-worker-row]").count(), 0);
          assert.equal(
            await page.evaluate(
              () => localStorage.length + sessionStorage.length,
            ),
            0,
          );
          await page.unroute("**/admin/stats", expire);
          await connect();
          await page.locator("#connection-button").click();
          await page.locator("#disconnect-button").click();
          assert.equal(await page.locator("[data-worker-row]").count(), 0);
          assert.equal(await page.locator("#admin-token").inputValue(), "");
          assert.equal((await page.content()).includes(token), false);
        },
      );

      assert.deepEqual(errors, [], "browser JavaScript errors");
      assert.deepEqual(cspErrors, [], "CSP violations");
      assert.deepEqual(external, [], "unexpected CDN/external requests");
      console.log(`Screenshots: ${artifacts}`);
    } finally {
      await context.close();
      await browser.close();
    }
  },
);
