// Render time per view (ms); the last two are legacy hashes that must still resolve.
const load = require("./load");
(async () => {
  const { w, errs } = await load(process.argv[2] || "../index.html", "#overview");
  const go = t => { w.location.hash = t; w.dispatchEvent(new w.HashChangeEvent("hashchange")); };
  for (const t of ["overview", "train/step", "train/par", "train/quant", "train/calib", "infer/step", "infer/deploy", "infer/quant", "infer/spec", "rl/mismatch", "rl/determinism", "rl/lowprec", "rl/qat", "appendix/data", "appendix/quant", "appendix/models", "appendix/method", "par/cp", "public/quant"]) { const t0 = Date.now(); go(t); console.log(t, Date.now() - t0); }
  console.log("ERRORS:", errs.length ? errs.join("\n") : "none");
})();
