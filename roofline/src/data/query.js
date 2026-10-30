/* 数据点库的查询接口。DB_RAW 由 tools/build_db.py 生成（src/data/db.js）。
   每个点 = {id, set, ...维度, ...指标}；数据集的 origin 是 "anon"（匿名实测）或 "public"（公开来源，点上带 src / url）。 */
const DB = (function(){
  const bySet = {};
  for (const p of DB_RAW.points) (bySet[p.set] = bySet[p.set] || []).push(p);
  const match = (p, f) => Object.keys(f).every(k => typeof f[k] === "function" ? f[k](p[k]) : Array.isArray(f[k]) ? f[k].includes(p[k]) : p[k] === f[k]);
  return {
    sets: DB_RAW.sets,
    meta: id => DB_RAW.sets[id],
    set: id => bySet[id] || [],
    where: (id, f) => (bySet[id] || []).filter(p => match(p, f || {})),
    one: (id, f) => (bySet[id] || []).find(p => match(p, f || {})),
    count: id => (bySet[id] || []).length,
    // 公开点显示出处，匿名点只显示“匿名数据点”
    origin: p => DB_RAW.sets[p.set].origin === "anon" ? "匿名数据点" : p.src,
  };
})();
