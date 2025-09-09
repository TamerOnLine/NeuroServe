from app.plugins.base import AIPlugin
from app.plugins.loader import get as get_plugin
import os, re, uuid, logging
from time import perf_counter
from datetime import datetime, timezone

log = logging.getLogger("workflow")
DEBUG = os.getenv("WORKFLOW_DEBUG", "0").lower() in ("1", "true", "yes")

# ---------- utils ----------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def pluck(obj, dotted, default=None):
    cur = obj
    for part in str(dotted).split("."):
        if part == "": 
            continue
        if isinstance(cur, list):
            try:
                part = int(part)
            except Exception:
                return default
        try:
            cur = cur[part]
        except (KeyError, IndexError, TypeError):
            return default
    return cur

_TMPL = re.compile(r"{{\s*([^}]+)\s*}}")
def render_tmpl(s: str, ctx: dict):
    def repl(m):
        key = m.group(1)
        val = pluck(ctx, key, "")
        return "" if val is None else str(val)
    return _TMPL.sub(repl, s)

def render_obj(x, ctx):
    if isinstance(x, str): 
        return render_tmpl(x, ctx)
    if isinstance(x, list): 
        return [render_obj(i, ctx) for i in x]
    if isinstance(x, dict): 
        return {k: render_obj(v, ctx) for k, v in x.items()}
    return x

def chunk_text(text: str, max_chars=1200, overlap=80):
    text = (text or "").replace("\r", "")
    out, n, i = [], len(text), 0
    while i < n:
        j = min(i + max_chars, n)
        # قص ذكي عند حدود سطر/جملة إن أمكن
        cut = max(text.rfind("\n", i, j), text.rfind(". ", i, j))
        if cut == -1 or cut <= i + 50:
            cut = j
        out.append(text[i:cut].strip())
        i = n if cut >= n else max(cut - overlap, 0)
    return [s for s in out if s]

def infer_local(provider: str, task: str, payload: dict) -> dict:
    plugin = get_plugin(provider)
    if plugin is None:
        return {"error": f"unknown provider '{provider}'", "payload": payload}
    req = dict(payload)
    req["task"] = task
    return plugin.infer(req)
# ---------- /utils ----------

class Plugin(AIPlugin):
    """
    provider: workflow
    task: run

    الخطوات تدعم:
      - op: infer | set | when | foreach | chunk  (الافتراضي: infer)
      - name: اسم الخطوة لمرجعتها لاحقًا
      - provider / task (للـ infer)
      - from: "stepName.path" لأخذ قيمة كنص دخل
      - text: نص يدعم {{vars.key}} و {{step.field}} وقيم من نتائج سابقة
      - params: تُدمج كما هي بعد التمبليت
      - when: if/then لتنفيذ مشروط
      - foreach: in/do/to للتكرار على قائمة
      - chunk: from/max_chars/overlap/to لتقطيع نص طويل
    """
    tasks = ["run"]

    def load(self) -> None:
        log.info("[plugin] workflow ready (sequential local-call)")

    def infer(self, payload: dict) -> dict:
        workflow_id = payload.get("workflow_id") or str(uuid.uuid4())
        steps = payload.get("steps") or []
        stop_on_error = bool(payload.get("stop_on_error", True))

        if not steps:
            return {"provider":"workflow","task":"run","error":"steps[] is required"}

        ctx = {"vars": {}}    # مساحة متغيرات عامة
        results = {}          # نتائج الخطوات بالاسم
        trace = []
        step_timings = []

        t0 = perf_counter()
        started_at = utc_now_iso()

        def run_steps(seq):
            nonlocal ctx, results, trace, step_timings
            for idx, step in enumerate(seq, start=1):
                op = step.get("op", "infer")
                name = step.get("name") or f"step{len(trace)+1}"

                s0 = perf_counter()
                s_started = utc_now_iso()
                out = {}

                try:
                    if op == "set":
                        key = step["key"]
                        val = render_obj(step.get("value"), {**results, **ctx})
                        ctx["vars"][key] = val
                        out = {"ok": True, "vars": {key: val}}

                    elif op == "chunk":
                        source = step.get("from")
                        text = pluck(results, source) if source else step.get("text", "")
                        text = render_tmpl(text or "", {**results, **ctx})
                        max_chars = int(step.get("max_chars", 1200))
                        overlap  = int(step.get("overlap", 80))
                        chunks = chunk_text(text, max_chars, overlap)
                        tgt = step.get("to", "chunks")
                        ctx["vars"][tgt] = chunks
                        out = {"chunks": len(chunks)}

                    elif op == "when":
                        cond = render_tmpl(step.get("if", ""), {**results, **ctx})
                        truthy = bool(cond) and cond.lower() not in ("0", "false", "none")
                        out = {"executed": False}
                        if truthy:
                            then_steps = step.get("then", [])
                            sub = run_steps(then_steps)
                            out = {"executed": True, "last": sub}

                    elif op == "foreach":
                        src = step.get("in")
                        items = pluck(ctx, f"vars.{src}") or pluck(results, src) or []
                        body = step.get("do", [])
                        collected = []
                        for i, item in enumerate(items):
                            ctx["vars"]["item"] = item
                            ctx["vars"]["index"] = i
                            sub = run_steps(body)
                            collected.append(sub)
                        tgt = step.get("to", "list")
                        ctx["vars"][tgt] = collected
                        out = {"count": len(collected)}

                    else:  # infer
                        provider = step.get("provider")
                        task = step.get("task")
                        if not provider or not task:
                            raise ValueError(f"step '{name}' missing provider/task")

                        payload_out = {}

                        # text من from أو من text مباشرة
                        if "from" in step:
                            payload_out["text"] = pluck(results, step["from"])
                        if "text" in step:
                            payload_out["text"] = render_tmpl(step["text"], {**results, **ctx})

                        # params بعد templating
                        params = render_obj(step.get("params", {}), {**results, **ctx})
                        payload_out.update(params)

                        # طباعة تتبّع قبل التنفيذ
                        if DEBUG:
                            txt_len = len(payload_out.get("text", "") or "") if isinstance(payload_out.get("text"), str) else "NA"
                            print(f"[workflow {workflow_id}] -> {provider}:{task} | step={idx}/{len(seq)} | name={name} | text.len={txt_len}")

                        out = infer_local(provider, task, payload_out)
                        results[name] = out

                    # تجميعة التتبع
                    trace.append({"name": name, "op": op, "step": step, "out": out})

                except Exception as e:
                    out = {"error": str(e)}
                    trace.append({"name": name, "op": op, "step": step, "out": out})
                    if stop_on_error:
                        e_ms = (perf_counter() - s0) * 1000
                        step_timings.append({
                            "name": name, "op": op, "started_at": s_started,
                            "ended_at": utc_now_iso(), "elapsed_ms": round(e_ms, 1), "error": str(e)
                        })
                        if DEBUG:
                            print(f"[workflow {workflow_id}] !! ERROR at {name}:{op} after {e_ms:.1f} ms -> {e}")
                        return out

                # زمن الخطوة
                s_ms = (perf_counter() - s0) * 1000
                step_timings.append({
                    "name": name, "op": op,
                    "started_at": s_started, "ended_at": utc_now_iso(),
                    "elapsed_ms": round(s_ms, 1)
                })
                if DEBUG:
                    print(f"[workflow {workflow_id}] <- {name}:{op} done in {s_ms:.1f} ms")

            return trace[-1]["out"] if trace else {}

        # نفّذ السلسلة
        last = run_steps(steps)

        ended_at = utc_now_iso()
        total_ms = (perf_counter() - t0) * 1000
        if DEBUG:
            print(f"[workflow {workflow_id}] TOTAL {total_ms:.1f} ms")

        # إخراج مختصر شائع إن وُجدت هذه الخطوات
        outputs = {
            "pdf_text": pluck(results, "pdf.output"),
            "summary": pluck(results, "sum.summary"),
            "sentiment": pluck(results, "cls.results.0"),
            "translation_de": pluck(results, "de.output"),
            "vars": ctx.get("vars", {})
        }

        return {
            "provider": "workflow",
            "task": "run",
            "workflow_id": workflow_id,
            "timing": {
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_ms": round(total_ms, 1),
                "steps": step_timings
            },
            "outputs": outputs,
            "steps": trace,
            "last": last
        }
