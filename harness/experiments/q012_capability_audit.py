#!/usr/bin/env python3
"""Offline Q012 capability qualification: compare raw receipts, model hashes, runtime logs.

Never promotes CLI advertised features to runtime PASS. Fails closed on missing raw evidence.
"""
from __future__ import annotations
from gates import demand
import argparse,datetime,hashlib,json,math,os,re
from pathlib import Path

# Preserve the original private-session layout, but allow an isolated mirror.
SRC=Path(os.environ.get("GNOSTRAL_Q012_EVIDENCE_ROOT", "/mnt/hdd/lab/sessions"))
RECEIPTS={
 "dense":SRC/"gnostral-q012-native-capabilities-retry2-20261010",
 "moe":SRC/"gnostral-q012-native-inference-capabilities-20261010",
 "embedding":SRC/"gnostral-q012-qwen3-embedding-complete-20261010",
}
NEGATIVE={
 "cli_argument_order":SRC/"gnostral-q012-native-inference-capabilities-20261010",
 "unsupported_context_override":SRC/"gnostral-q012-native-capabilities-retry-20261010",
 "bert_loader_unsupported":SRC/"gnostral-q012-native-capabilities-retry2-20261010",
 "incomplete_qwen3_modules":SRC/"gnostral-q012-qwen3-embedding-20261010",
 "embedding_cacheless_engine_startup":SRC/"gnostral-q012-qwen3-embedding-complete-20261010",
}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1048576*4),b""):h.update(chunk)
 return h.hexdigest()
def main()->None:
 ap=argparse.ArgumentParser()
 ap.add_argument("--all-pass-session",type=Path,help="Single session proving all families on the same native binary")
 ap.add_argument("--embedding-pass-session",type=Path)
 ap.add_argument("--embedding-pass-binary",type=Path)
 ap.add_argument("--rehash-models",action="store_true",help="Revalidate locally accessible source model weight hashes")
 ap.add_argument("--output",type=Path,required=True)
 a=ap.parse_args()
 if a.output.exists():raise SystemExit("refuse overwrite")
 documents={}
 rehashed_models={}
 for profile,session in RECEIPTS.items():
  if a.all_pass_session:
   session=a.all_pass_session
  elif profile=="embedding" and a.embedding_pass_session:
   session=a.embedding_pass_session
  r=json.loads((session/f"{profile}-receipt.json").read_text())
  f=json.loads((session/"frozen-protocol.json").read_text())
  demand(isinstance(f.get("runtime_binary_sha256"),str) and len(f["runtime_binary_sha256"])==64, "gate at q012_capability_audit.py:47")
  log=(session/f"{profile}-server.log").read_text(errors="replace")
  demand(r["server_reaped"] is True, "gate at q012_capability_audit.py:49")
  demand(abs(r["gpu_ambient_reclaim_delta_mib"])<=128, "gate at q012_capability_audit.py:50")
  demand(r.get("telemetry",{}).get("peak_total_gpu_used_mib",0)>0, "gate at q012_capability_audit.py:51")
  demand(session.joinpath("summary.json").is_file(), "gate at q012_capability_audit.py:52")
  summary=json.loads((session/"summary.json").read_text())
  demand(summary["engine_binary_sha256"]==f["runtime_binary_sha256"], "gate at q012_capability_audit.py:54")
  demand(summary["entries"][profile]["status"]==r["status"], "gate at q012_capability_audit.py:55")
  for model,weight in f["weight_sha256"].items():
   demand(len(weight)==64, "gate at q012_capability_audit.py:57")
   if a.rehash_models:
    source=Path(f["source_snapshot"][model])
    source=source/"model.safetensors" if source.is_dir() else source
    if source not in rehashed_models:
     rehashed_models[source]=sha(source)
    demand(rehashed_models[source]==weight,(model,"model weight source has changed"))
  if r["status"]=="PASS_FUNCTIONAL":
   probe=r["probe"]
   if profile=="dense":
    demand(probe["answer"].strip()=="GNOSTRAL", "gate at q012_capability_audit.py:67")
    demand("Text" in log, "gate at q012_capability_audit.py:68")
   elif profile=="moe":
    demand("MoE" in probe["answer"] or "experts" in probe["answer"], "gate at q012_capability_audit.py:70")
    demand("mixed-device" in log and "disabling PagedAttention" in log, "gate at q012_capability_audit.py:71")
    demand("native expert residency mixed-device admission" in log, "MoE native residency not compiled/activated")
    demand("native expert residency engine counters initialized" in log, "MoE residency counters absent")
   else:
    demand(probe["kind"]=="embedding.vector", "gate at q012_capability_audit.py:75")
    demand(probe["n"]==3 and probe["dimension"]==1024, "gate at q012_capability_audit.py:76")
    demand(all(math.isfinite(v) and v>0 for v in probe["norms"]), "gate at q012_capability_audit.py:77")
    demand(probe["related_higher"] is True, "gate at q012_capability_audit.py:78")
    demand("Model loaded." in log, "gate at q012_capability_audit.py:79")
    demand("🔢 Embedding" in log, "gate at q012_capability_audit.py:80")
    payload=json.loads((session/"embedding-vectors-private.json").read_text())
    vec=payload["vectors"]
    demand(len(vec)==3 and all(len(item)==1024 for item in vec), "gate at q012_capability_audit.py:83")
    demand(all(all(isinstance(v,(int,float)) and math.isfinite(v) for v in item) for item in vec), "gate at q012_capability_audit.py:84")
    norm=[math.sqrt(sum(v*v for v in item)) for item in vec]
    def cosine(a,b,na,nb):
     return sum(x*y for x,y in zip(a,b))/(na*nb)
    related=cosine(vec[0],vec[1],norm[0],norm[1])
    unrelated=cosine(vec[0],vec[2],norm[0],norm[2])
    demand(related>unrelated, "gate at q012_capability_audit.py:90")
    demand(all(abs(x-y)<1e-5 for x,y in zip(norm,probe["norms"])), "gate at q012_capability_audit.py:91")
    demand(abs(related-probe["related_cosine"])<1e-5, "gate at q012_capability_audit.py:92")
    demand(abs(unrelated-probe["unrelated_cosine"])<1e-5, "gate at q012_capability_audit.py:93")
    for i,v in enumerate(vec):
     reported=json.loads((session/"embedding-response.json").read_text())["items"][i]["sha256"]
     demand(hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()==reported, "gate at q012_capability_audit.py:96")
  documents[profile]={
   "status":r["status"],"binary_sha256":f["runtime_binary_sha256"],
   "model_weight_sha256":f["weight_sha256"][profile],
   "session":str(session),
   "receipt_sha256":sha(session/f"{profile}-receipt.json"),
   "log_sha256":sha(session/f"{profile}-server.log"),
   "response_sha256":sha(session/f"{profile}-response.json") if (session/f"{profile}-response.json").exists() else None,
   "private_raw_vector_sha256":sha(session/"embedding-vectors-private.json") if profile=="embedding" and (session/"embedding-vectors-private.json").exists() else None,
   "gpu_peak_total_mib":r.get("telemetry",{}).get("peak_total_gpu_used_mib"),
   "rss_peak_mib":r.get("telemetry",{}).get("peak_owned_process_tree_rss_mib"),
   "reclaim_delta_mib":r["gpu_ambient_reclaim_delta_mib"],
   "probe":r.get("probe"),
  }
 for name,session in NEGATIVE.items():
  target={
   "cli_argument_order":"dense",
   "unsupported_context_override":"dense",
   "bert_loader_unsupported":"embedding",
   "incomplete_qwen3_modules":"embedding",
   "embedding_cacheless_engine_startup":"embedding",
  }[name]
  r=json.loads((session/f"{target}-receipt.json").read_text())
  demand(r["status"].startswith("NOT_QUALIFIED"),(name,r["status"]))
  failure_log=(session/f"{target}-server.log").read_text(errors="replace")
  expected_witness={
   "cli_argument_order":"unexpected argument '--host'",
   "unsupported_context_override":"max_model_len",
   "bert_loader_unsupported":"Unsupported Hugging Face Transformers model class",
   "incomplete_qwen3_modules":"embedding.rs:495",
   "embedding_cacheless_engine_startup":"embedding.rs:789",
  }[name]
  demand(expected_witness in failure_log,(name,"expected cause not witnessed"))
  documents.setdefault("negative",{})[name]={
   "status":r["status"],"session":str(session),
   "receipt_sha256":sha(session/f"{target}-receipt.json"),
   "log_sha256":sha(session/f"{target}-server.log"),
   "error_excerpt":"; ".join((session/f"{target}-server.log").read_text(errors="replace").splitlines()[-4:])[:560]
  }
 if a.embedding_pass_binary:
  demand(sha(a.embedding_pass_binary)==documents["embedding"]["binary_sha256"],"new binary hash mismatch")
 profiles=documents
 v={"schema":"gnostral.q012.capability-audit.v1",
  "generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "scope":"sequential isolated fresh server per profile; SAME binary required for full convergence",
  "profiles":profiles,
  "same_binary_dense_moe":profiles["dense"]["binary_sha256"]==profiles["moe"]["binary_sha256"],
  "single_binary_3_profiles":len({profiles[k]["binary_sha256"] for k in ("dense","moe","embedding")})==1 and all(profiles[k]["status"]=="PASS_FUNCTIONAL" for k in ("dense","moe","embedding")),
  "all_servers_reaped":True,
  "not_claimed":["same long-lived process","orchestration","hot_swap","pagedattention","production stable","high quality embedding retrieval"],
 }
 v["verdict"]="THREE_FAMILIES_SAME_BINARY_PASS" if v["single_binary_3_profiles"] else "PARTIAL_Q012_QUALIFICATION"
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 print(v["verdict"],[(k,v["profiles"][k]["status"]) for k in ("dense","moe","embedding")])
if __name__=="__main__":
 main()
