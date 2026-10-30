// Load the multi-file page in jsdom with its local scripts; resolves once every script has run.
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");
module.exports = async function load(page, hash) {
  const errs = [];
  const vc = new VirtualConsole();
  vc.on("error", e => errs.push(String(e && e.stack || e)));
  vc.on("jsdomError", e => errs.push("jsdomError: " + (e.stack || e.message)));
  const dom = await JSDOM.fromFile(path.resolve(page), { runScripts: "dangerously", resources: "usable", virtualConsole: vc, pretendToBeVisual: true,
    beforeParse(w) { w.Element.prototype.scrollIntoView = function () {}; if (hash) w.location.hash = hash; } });
  await new Promise(r => dom.window.addEventListener("load", r));
  return { dom, w: dom.window, d: dom.window.document, errs };
};
