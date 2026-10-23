#!/usr/bin/env python3
from gates import demand
from pathlib import Path
import json,hashlib,statistics,datetime,re,subprocess
S=Path("/mnt/hdd/lab/sessions/gnostral-q010-cpu-moe-hotpath-retest-20261010")
F=Path("/mnt/hdd/lab/sessions/gnostral-q010-cpu-moe-hotpath-20261010")
REPO=Path("/mnt/sandbox/sessions/gnostral-q010-cpu-moe-hotpath-20261010/repo")
MODEL=Path("/mnt/models/active/gnostral/Qwen3-30B-A3B-Q2_K.gguf")
BASE=Path("/home/tension_atoi/Projects/gnostral.rs/vendor/gnostral-strata-worktree/target-strata-candidate-release/release/mistralrs")
CAND=Path("/mnt/sandbox/sessions/gnostral-q010-cpu-moe-hotpath-20261010/target/release/mistralrs")
PROFILE=Path("/home/tension_atoi/Projects/gnostral.rs/vendor/gnostral-strata-worktree/harness/profiles/qwen3-30b-a3b-q2k-deterministic-4-per-layer.json")
def h(p):
 dig=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(4*1024*1024),b""):dig.update(x)
 return dig.hexdigest()
p=json.loads((S/"frozen-protocol.json").read_text())
current=json.loads((S/"summary.json").read_text())
previous=json.loads((F/"summary.json").read_text())
contract=json.loads((S/"RETEST-CONTRACT.json").read_text())
demand(current["status"]=="ALL_SIX_TRIALS_PASS" and current["completed"]==6 and current["by_engine"]=={"baseline":3,"candidate":3}, "gate at q010_hotpath_ab_audit_20261010.py:20")
demand(previous["status"]=="NOT_QUALIFIED" and previous["completed"]==2, "gate at q010_hotpath_ab_audit_20261010.py:21")
demand(p["repetition_order"]==["baseline","candidate","candidate","baseline","baseline","candidate"], "gate at q010_hotpath_ab_audit_20261010.py:22")
gate=p["predeclared_success_gate"]
demand(gate["candidate_median_tps_ratio_minimum"]==1.1 and gate["max_allowed_increase_gpu_peak_mib"]==128 and gate["max_allowed_increase_host_rss_mib"]==512, "gate at q010_hotpath_ab_audit_20261010.py:24")
demand(h(MODEL)==p["source_sha256"]["model"], "gate at q010_hotpath_ab_audit_20261010.py:25")
demand(h(BASE)==p["source_sha256"]["strata"], "gate at q010_hotpath_ab_audit_20261010.py:26")
demand(h(CAND)=="dc11a9ad45574fea0565768bb877cd828b4ec7ee9eb1177537119959c42984c3", "gate at q010_hotpath_ab_audit_20261010.py:27")
demand(h(PROFILE)==p["source_sha256"]["profile"], "gate at q010_hotpath_ab_audit_20261010.py:28")
demand(h(F/"candidate_ab.py")==contract["source_script_sha256"], "gate at q010_hotpath_ab_audit_20261010.py:29")
demand(h(S/"candidate_ab_retest.py")==contract["retest_script_sha256"], "gate at q010_hotpath_ab_audit_20261010.py:30")
demand(contract["only_source_change"]=="S path to distinct lab session", "gate at q010_hotpath_ab_audit_20261010.py:31")
demand(subprocess.check_output(["git","-C",str(REPO),"rev-parse","--short","HEAD"],text=True).strip()=="67040aa", "gate at q010_hotpath_ab_audit_20261010.py:32")
prior=json.loads((REPO/"evidence/runs/q010-cpu-hotpath-parity-20261010.json").read_text())
demand(prior["candle_tests"]["passed"]==6 and prior["mistralrs_cpu_route"]["passed"]==1, "gate at q010_hotpath_ab_audit_20261010.py:34")
rows=[];digests={};paired=[]
for e in ("baseline","candidate"):
 for i in (1,2,3):
  name=f"{i:02d}-{e}"
  f=S/(name+".json")
  r=json.loads(f.read_text())
  demand(r["status"]=="PASS_SAMPLES" and r["server_reaped"] is True, "gate at q010_hotpath_ab_audit_20261010.py:41")
  demand(r["engine"]==e and r["repeat"]==i and abs(r["reclaim_delta_mib"])<=128, "gate at q010_hotpath_ab_audit_20261010.py:42")
  demand(r["before"]["ram_available_mib"]>=16384 and r["before"]["gpu"]["free_mib"]>=1536, "gate at q010_hotpath_ab_audit_20261010.py:43")
  demand(r["ctx_requested"]==1024 and r["telemetry"]["samples"]>10, "gate at q010_hotpath_ab_audit_20261010.py:44")
  demand(Path(r["server_start_cmd_redacted"][0])==(BASE if e=="baseline" else CAND), "gate at q010_hotpath_ab_audit_20261010.py:45")
  digests[f.name]=h(f)
  digests[name+".server.log"]=h(S/(name+".server.log"))
  for budget in (128,512):
   measure=r["samples"][str(budget)]
   output=S/(name+f"-{budget}-response.txt")
   demand(h(output)==measure["output_sha256"], "gate at q010_hotpath_ab_audit_20261010.py:51")
   demand(0<measure["tokens"]<budget and measure["tps_engine"]>0, "gate at q010_hotpath_ab_audit_20261010.py:52")
   txt=output.read_text()
   if budget==512:
    demand(all(f"{k}." in txt for k in range(1,7)), "gate at q010_hotpath_ab_audit_20261010.py:55")
   else:
    demand(txt.count(".")>=2, "gate at q010_hotpath_ab_audit_20261010.py:57")
   digests[output.name]=h(output)
   rows.append({
    "engine":e,"repeat":i,"budget":budget,"generated_tokens":measure["tokens"],
    "decode_tps":measure["tps_engine"],"prefill_ms":measure["prefill_ms_engine"],
    "hot_stream_ttft_ms":measure.get("separate_request_stream",{}).get("first_text_ms") if budget==128 else None,
    "gpu_peak_total_mib":r["telemetry"]["peak_total_gpu_used_mib"],
    "host_rss_peak_mib":r["telemetry"]["peak_owned_process_tree_rss_mib"],
    "gpu_ambient_before_mib":r["before"]["gpu"]["used_mib"],
    "post_reclaim_delta_mib":r["reclaim_delta_mib"],
    "response_sha256":measure["output_sha256"]
   })
for budget in (128,512):
 for i in (1,2,3):
  a=(S/f"{i:02d}-baseline-{budget}-response.txt").read_bytes()
  b=(S/f"{i:02d}-candidate-{budget}-response.txt").read_bytes()
  demand(a==b, "gate at q010_hotpath_ab_audit_20261010.py:73")
  paired.append({"repeat":i,"budget":budget,"identical_bytes":True,"sha256":hashlib.sha256(a).hexdigest()})
agg={}
for e in ("baseline","candidate"):
 own=[x for x in rows if x["engine"]==e]
 def med(k):return statistics.median(x[k] for x in own if x["budget"]==128)
 agg[e]={
  "median_tps":{str(b):statistics.median(x["decode_tps"] for x in own if x["budget"]==b) for b in (128,512)},
  "median_prefill_ms":{str(b):statistics.median(x["prefill_ms"] for x in own if x["budget"]==b) for b in (128,512)},
  "median_hot_stream_ttft_ms":med("hot_stream_ttft_ms"),
  "median_peak_gpu_total_mib":med("gpu_peak_total_mib"),
  "median_host_rss_peak_mib":med("host_rss_peak_mib"),
  "max_absolute_reclaim_delta_mib":max(abs(x["post_reclaim_delta_mib"]) for x in own)
 }
 for b in ("128","512"):
  demand(abs(agg[e]["median_tps"][b]-current["median_decode_tps"][e][b])<1e-9, "gate at q010_hotpath_ab_audit_20261010.py:88")
 demand(agg[e]["median_peak_gpu_total_mib"]==current["median_gpu_peak_total_mib"][e], "gate at q010_hotpath_ab_audit_20261010.py:89")
 demand(abs(agg[e]["median_host_rss_peak_mib"]-current["median_owned_rss_peak_mib"][e])<1e-9, "gate at q010_hotpath_ab_audit_20261010.py:90")
ratios={b:agg["candidate"]["median_tps"][b]/agg["baseline"]["median_tps"][b] for b in ("128","512")}
gpu_delta=agg["candidate"]["median_peak_gpu_total_mib"]-agg["baseline"]["median_peak_gpu_total_mib"]
rss_delta=agg["candidate"]["median_host_rss_peak_mib"]-agg["baseline"]["median_host_rss_peak_mib"]
demand(ratios["512"]>=1.1 and gpu_delta<=128 and rss_delta<=512 and len(paired)==6, "gate at q010_hotpath_ab_audit_20261010.py:94")
for fn in ("frozen-protocol.json","summary.json","RETEST-CONTRACT.json","candidate_ab_retest.py","retest-controller.log"):
 digests[fn]=h(S/fn)
kernel=subprocess.run(["journalctl","-k","--since","2026-10-10 02:54:00","--until","2026-10-10 03:01:00","-o","cat","--no-pager"],capture_output=True,text=True,timeout=30)
driver={"journal_rc":kernel.returncode,
 "NVRM_or_Xid":[l for l in kernel.stdout.splitlines() if re.search("NVRM|Xid",l,re.I)][:15],
 "scope":"host kernel journal only, not per-process NVIDIA driver attestation"}
driver["pass"]=(kernel.returncode==0 and not driver["NVRM_or_Xid"])
demand(driver["pass"], "gate at q010_hotpath_ab_audit_20261010.py:102")
audit={
 "schema":"q010-cpu-borrowed-q2kq3k-ab-independent-audit-v1",
 "status":"REPEATED_AB_PASS_INITIAL_FAILURE_RETAINED",
 "time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "failed_first_attempt":{"path":str(F),"status":previous["status"],"completed":2,
  "candidate_post_reclaim_gpu_delta_mib":196,"root_cause":"undetermined, potentially desktop GPU traffic"},
 "retest_session":str(S),"process_runs":6,"measured_responses":12,
 "model_sha256_post_run":h(MODEL),
 "binary_sha256":{"baseline":h(BASE),"candidate":h(CAND)},
 "source_commit":"67040aa",
 "source_parity_receipt_sha256":h(REPO/"evidence/runs/q010-cpu-hotpath-parity-20261010.json"),
 "predeclared_gates":gate,
 "aggregate":agg,"candidate_over_baseline_median_tps_ratios":ratios,
 "candidate_minus_baseline_gpu_total_peak_mib":gpu_delta,
 "candidate_minus_baseline_host_rss_peak_mib":rss_delta,
 "paired_byte_identical_responses":paired,"host_driver_evidence":driver,
 "raw_artifacts_sha256":digests,"per_request":rows,
 "limitations":[
  "only 3 processes per engine; descriptive, not general model speed guarantee",
  "first attempt failed post-release GPU ambient gate and is retained without changing thresholds",
  "third paired 512-token response differs from first two but remains identical within its own baseline/candidate pair",
  "source unit tests prove Q2K/Q3K numerical parity on controlled cases, not exhaustive unsafe memory correctness",
  "per-route copy count, PSS, PCIe transfer volume and allocation profiling not recorded",
  "RSS includes mapped/shared pages, GPU total includes graphics desktop and sampling may miss short peaks",
  "TTFT measured on separate warm request, not identical cold inference",
  "no product, modeld, production, remote push or docs publication authorized"
 ]}
f=S/"independent-audit.json"
demand(not f.exists(), "gate at q010_hotpath_ab_audit_20261010.py:131")
f.write_text(json.dumps(audit,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
print("AUDIT_PASS",audit["status"],"ratios",ratios,"memory_deltas",gpu_delta,rss_delta)
print("PAIRED_RESPONSES",len(paired),"FILES_HASHED",len(digests),"DRIVER_WINDOW",driver["pass"])
print("RECEIPT",f)
