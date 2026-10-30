"""构建 roofline 的数据点库。

两类来源，合成一个库：
  - 匿名数据点：data/anon/*.json，每个文件一个数据集（测量条件、维度、指标和点）。只有条件和数字，不带出处。
  - 公开数据点：data/public/*.json，手工整理，每个点带 src / url。

输出：
  data/db.json        规范版本（每个点一行，方便 diff）
  src/data/db.js      页面加载的同一份数据（const DB_RAW = ...）
  --sqlite PATH       可选，导出成 SQLite 方便临时查询（sets / points 两张表，每个数据集一个视图）

用法：
  uv run tools/build_db.py                 # 重新生成 db.json 和 db.js
  uv run tools/build_db.py --check         # 只检查：生成结果和仓库里的是否一致
  uv run tools/build_db.py --sqlite /tmp/roofline.sqlite
"""
import argparse, json, pathlib, re, sqlite3, sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ANON_KEYS = ["title", "stage", "hw", "gpus", "topo", "model", "conditions", "dims", "metrics"]

PUBLIC = {
 "quant_evidence": dict(file="quant_evidence.json", stage="quant", title="量化：公开的 PR、博客与文档",
   conditions="权重 = 时效（半衰期 12 个月；只知道读取日期时 × 0.7）× 硬件匹配 × 来源类型。"),
 "inference_public": dict(file="inference.json", stage="infer", title="推理：公开的服务数字", conditions="各家公布的口径不同，只作量级参考。"),
 "training_public": dict(file="training.json", stage="train", title="训练：公开的 EP 后端数字", conditions="公布方自己的模型和集群。"),
 "spec_public": dict(file="spec.json", stage="infer", title="投机解码：公开数字", conditions="模型、温度、数据集和并发各不相同，只看量级和方向。"),
 "determinism_public": dict(file="determinism.json", stage="rl", title="确定性推理与训推逐位一致：公开数字",
   conditions="模型、硬件和测法各不相同；开销是各家相对自己基线的数字，只看量级。"),
 "releases": dict(file="releases.json", stage="meta", title="过去一年的发布与博客",
   conditions="LMSYS 博客，SGLang / vLLM / slime / Miles / AReaL 的 GitHub release；只摘和推理、训练、低精度、投机解码、RL 相关的要点。"),
}

ARCH_PERF = {"bw": 1.0, "bw120": 0.7, "hopper": 0.5, "amd": 0.5, "any": 0.9}
KIND = {"PR": 1.0, "blog": 0.9, "doc": 0.9}
def evidence_weight(p, today):
    months = (today - date.fromisoformat(p["date"])).days / 30.44
    rec = 0.5 ** (months / 12) * (0.7 if p["dk"] == "read" else 1.0)
    hw = 0.9 if p["nature"] == "fact" else (1.0 if p["arch"] in ("bw", "any") else 0.8) if p["nature"] in ("quality", "bug") else ARCH_PERF[p["arch"]]
    return round(rec * hw * KIND[p["kind"]], 3)

SANITIZE = re.compile(r"integration|internal|local\b|\+local|hostname|private|/home/|笔记|note", re.I)

def build(today):
    sets, points = {}, []
    for f in sorted((DATA / "anon").glob("*.json")):
        s = json.loads(f.read_text(encoding="utf-8")); sid = f.stem
        sets[sid] = dict({"title": s["title"], "origin": "anon"}, **{k: s.get(k) for k in ANON_KEYS[1:]})
        for p in s["points"]:
            points.append(dict({"set": sid}, **p, id=sid + ":" + "/".join(str(p.get(k)) for k in s["dims"])))
    for sid, s in PUBLIC.items():
        sets[sid] = {"title": s["title"], "origin": "public", "stage": s["stage"], "conditions": s["conditions"]}
        if sid == "quant_evidence": sets[sid]["as_of"] = today.isoformat()
        for p in json.load(open(DATA / "public" / s["file"], encoding="utf-8")):
            p = dict(p, set=sid)
            if sid == "quant_evidence":
                p["w"] = evidence_weight(p, today); p.setdefault("id", sid + ":" + p["src"] + ":" + p["claim"][:40])
            elif sid == "releases": p["id"] = sid + ":" + p["project"] + ":" + p["version"]
            else: p["id"] = sid + ":" + p["id"]
            points.append(p)
    return {"schema": 1, "sets": sets, "points": points}

def validate(db):
    errs, ids = [], set()
    for p in db["points"]:
        s = db["sets"].get(p["set"])
        if not s: errs.append("未知数据集 " + p["set"]); continue
        if p["id"] in ids: errs.append("重复 id " + p["id"])
        ids.add(p["id"])
        if s["origin"] == "anon":
            extra = set(p) - {"set", "id"} - set(s["dims"]) - set(s["metrics"])
            if extra: errs.append("匿名点有维度 / 指标以外的字段 %s：%s" % (sorted(extra), p["id"]))
            for k in s["dims"]:
                if k not in p: errs.append("缺维度 %s：%s" % (k, p["id"]))
        elif not p.get("src") and not p.get("url"): errs.append("公开点缺 src / url " + p["id"])
    for sid, s in db["sets"].items():
        if s["origin"] == "anon" and SANITIZE.search(json.dumps(s, ensure_ascii=False)): errs.append("匿名数据集的描述里有不该出现的词：" + sid)
        if not any(p["set"] == sid for p in db["points"]): errs.append("空数据集 " + sid)
    return errs

def dump_json(db):
    sets = json.dumps(db["sets"], ensure_ascii=False, indent=1)
    pts = ",\n".join(json.dumps(p, ensure_ascii=False, separators=(",", ":")) for p in db["points"])
    return '{"schema":%d,\n"sets":%s,\n"points":[\n%s\n]}\n' % (db["schema"], sets, pts)

def dump_js(db):
    return "/* 生成文件，不要手改：uv run tools/build_db.py（源数据在 data/） */\nconst DB_RAW = " + json.dumps(db, ensure_ascii=False, separators=(",", ":")) + ";\n"

def to_sqlite(db, path):
    path = pathlib.Path(path); path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript("""CREATE TABLE sets(id TEXT PRIMARY KEY, title TEXT, origin TEXT, stage TEXT, hw TEXT, gpus INT, topo TEXT, model TEXT, conditions TEXT, meta JSON);
      CREATE TABLE points(id TEXT PRIMARY KEY, set_id TEXT REFERENCES sets(id), data JSON);""")
    for sid, s in db["sets"].items():
        con.execute("INSERT INTO sets VALUES(?,?,?,?,?,?,?,?,?,?)", (sid, s["title"], s["origin"], s["stage"], s.get("hw"), s.get("gpus"), s.get("topo"), s.get("model"), s["conditions"], json.dumps(s, ensure_ascii=False)))
    con.executemany("INSERT INTO points VALUES(?,?,?)", [(p["id"], p["set"], json.dumps(p, ensure_ascii=False)) for p in db["points"]])
    for sid, s in db["sets"].items():
        cols = (s.get("dims") or []) + list(s.get("metrics") or {})
        if not cols: cols = sorted({k for p in db["points"] if p["set"] == sid for k in p} - {"id", "set"})
        sel = ", ".join("json_extract(data, '$.%s') AS \"%s\"" % (c, c) for c in cols)
        con.execute('CREATE VIEW "%s" AS SELECT id, %s FROM points WHERE set_id = %s' % (sid, sel, "'" + sid + "'"))
    con.commit(); con.close()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--sqlite")
    ap.add_argument("--today", help="量化证据权重的参照日期；默认今天，--check 时沿用 db.json 里的 as_of")
    a = ap.parse_args()
    today = a.today
    if not today and a.check and (DATA / "db.json").exists():
        today = json.loads((DATA / "db.json").read_text())["sets"]["quant_evidence"].get("as_of")
    db = build(date.fromisoformat(today or date.today().isoformat()))
    errs = validate(db)
    if errs: print("\n".join(errs)); sys.exit(1)
    out = {DATA / "db.json": dump_json(db), ROOT / "src" / "data" / "db.js": dump_js(db)}
    if a.check:
        stale = [str(p.relative_to(ROOT)) for p, t in out.items() if not p.exists() or p.read_text() != t]
        print("过期：" + ", ".join(stale) if stale else "db 是最新的"); sys.exit(1 if stale else 0)
    for p, t in out.items(): p.write_text(t)
    if a.sqlite: to_sqlite(db, a.sqlite)
    by = {}
    for p in db["points"]: by[p["set"]] = by.get(p["set"], 0) + 1
    print("%d 个点，%d 个数据集" % (len(db["points"]), len(db["sets"])))
    for sid, n in by.items(): print("  %-18s %-6s %4d  %s" % (sid, db["sets"][sid]["origin"], n, db["sets"][sid]["title"]))

if __name__ == "__main__":
    main()
