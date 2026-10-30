// For every model × framework × train preset print step KPIs (plus infer KPIs), flag NaN/undefined in each pane and script errors.
// Presets that carry their own sequence length / topology run in a second pass so they don't leak into the first.
const load = require("./load");
(async () => {
  const { w, d, errs } = await load(process.argv[2] || "../index.html", "#train/step");
  const go = t => { w.location.hash = t; w.dispatchEvent(new w.HashChangeEvent("hashchange")); };
  const setSel = (sel, v) => { const el = d.querySelector(sel); if (!el) return; el.value = v; el.dispatchEvent(new w.Event("change", { bubbles: true })); };
  const kp = id => Array.from(d.querySelectorAll(`#${id} .kpi`)).slice(0, 3).map(k => k.querySelector(".n").textContent).join(" | ");
  const warn = () => { const g = d.getElementById("g-warn"); return g.hidden ? "" : g.textContent.slice(0, 140); };
  const bad = key => { const p = d.querySelector(`section[data-pane="${key}"]`); return ((p.textContent + p.innerHTML).match(/(?<!得到 )NaN|undefined|Infinity/g) || []).length + (warn().includes("计算出错") ? 1000 : 0); };
  const VIEWS = ["overview", "train/quant", "train/calib", "infer/deploy", "infer/quant", "infer/spec", "rl/mismatch", "rl/lowprec", "rl/qat", "appendix/data", "appendix/quant", "appendix/models", "appendix/method"];
  const NEW = new Set(["h200cp8", "uly256", "mok512", "dm128", "deepep64"]);
  const t0 = Date.now();
  for (const pass of [0, 1]) for (const m of ["qwen35", "qwen38_27b", "glm53f", "musespark", "kimik3"]) {
    setSel("#g-model", m);
    for (const fw of ["megatron", "torchtitan"]) {
      go("train/step"); setSel("[data-bind='trn.fw']", fw);
      const ids = Array.from(d.querySelectorAll("#t-preset option")).map(o => o.value).filter(v => v !== "custom" && NEW.has(v) === !!pass);
      for (const id of ids) {
        setSel("#t-preset", id); go("train/step");
        console.log("TRN", m, fw, id, d.getElementById("g-topo").value, d.getElementById("g-hw").value, "|", kp("t-kpis"), "|", bad("train/step"), warn());
        go("train/par"); const b = bad("train/par"); if (b) console.log("  PAR BAD", b, warn());
      }
    }
    if (pass) continue;
    go("infer/step"); console.log("INF", m, kp("i-kpis"), bad("infer/step"), warn());
    for (const t of VIEWS) { go(t); const b = bad(t); if (b) console.log("  BAD", t, b, warn()); }
  }
  console.log("time", Date.now() - t0, "ms");
  console.log("ERRORS:", errs.length ? errs.slice(0, 5).join("\n") : "none");
  process.exit(errs.length ? 1 : 0);
})();
