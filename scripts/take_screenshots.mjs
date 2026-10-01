// Capture UI screenshots for docs/screenshots with headless Chrome (puppeteer).
//
//   npm i -g puppeteer            (or: npm i puppeteer in any folder and set NODE_PATH)
//   WEB_URL=http://localhost:3000 node scripts/take_screenshots.mjs
//
// Logs in through the real API for each role, then drives the real UI.
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const puppeteer = require("puppeteer");

const WEB = process.env.WEB_URL || "http://localhost:3000";
const OUT = process.env.OUT_DIR || new URL("../docs/screenshots/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
mkdirSync(OUT, { recursive: true });

async function login(page, username) {
  const res = await fetch(`${WEB}/api/auth/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ username, password: "demo" }),
  });
  const { access_token, user } = await res.json();
  await page.goto(`${WEB}/login`, { waitUntil: "networkidle0" });
  await page.evaluate(
    (t, u) => {
      localStorage.setItem("novapromo.token", t);
      localStorage.setItem("novapromo.user", JSON.stringify(u));
    },
    access_token,
    user,
  );
}

async function chat(page, username, message, file) {
  await login(page, username);
  await page.goto(`${WEB}/chat`, { waitUntil: "networkidle0" });
  await page.type(".chat-input textarea", message);
  const done = page.waitForResponse((r) => r.url().includes("/api/chat") && r.request().method() === "POST");
  await page.click(".chat-input button");
  await done;
  await new Promise((r) => setTimeout(r, 600));
  // Sticky header/sidebar + page scrolling: capture the viewport from the top of the conversation.
  await page.evaluate(() => window.scrollTo(0, 0));
  await new Promise((r) => setTimeout(r, 200));
  await page.screenshot({ path: `${OUT}${file}` });
  console.log("saved", file);
}

async function visit(page, username, path, file, action) {
  await login(page, username);
  await page.goto(`${WEB}${path}`, { waitUntil: "networkidle0" });
  if (action) await action(page);
  await new Promise((r) => setTimeout(r, 800));
  await page.screenshot({ path: `${OUT}${file}`, fullPage: true });
  console.log("saved", file);
}

const browser = await puppeteer.launch({ headless: true, defaultViewport: { width: 1440, height: 1300 } });
const page = await browser.newPage();
try {
  await chat(page, "samuel.sales", "Find black metal bottles around 750 ml under 15 EUR that are in stock.", "01-chat-hybrid-search.png");
  await chat(page, "samuel.sales", "What is our policy for urgent supplier deliveries?", "02-chat-rag-citations.png");
  await chat(page, "eva.employee", "What is our policy for urgent supplier deliveries?", "03-chat-rag-rbac-employee.png");
  await chat(page, "samuel.sales", "How many pieces of SKU NP-10042 are currently available?", "04-chat-sql-tool.png");
  await chat(page, "katarina.catalog", "Generate a new Slovak description for NP-10042.", "05-chat-content-agent-approval.png");
  await chat(page, "eva.employee", "Summarize the latest supplier notice from EcoLine.", "06-chat-prompt-injection.png");
  await visit(page, "samuel.sales", "/search", "07-hybrid-search-explain.png", async (p) => {
    const box = 'input[placeholder^="Natural-language"]';
    await p.click(box, { clickCount: 3 });
    await p.keyboard.press("Backspace");
    await p.type(box, "black metal bottle 750 ml under 15 EUR");
    const done = p.waitForResponse((r) => r.url().includes("/api/search/products"));
    await p.keyboard.press("Enter");
    await done;
  });
  await visit(page, "adam.admin", "/products/NP-10042", "08-product-detail.png");
  await visit(page, "adam.admin", "/approvals", "09-approvals.png");
  await visit(page, "adam.admin", "/batch", "10-batch-jobs.png");
  await visit(page, "adam.admin", "/activity", "11-agent-activity.png", async (p) => {
    const row = await p.$("tbody tr");
    if (row) await row.click();
  });
  await visit(page, "adam.admin", "/system", "12-system-metrics.png");
} finally {
  await browser.close();
}
