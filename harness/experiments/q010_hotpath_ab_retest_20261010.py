#!/usr/bin/env python3
"""Q-009 128/512-token ecosystem comparison; owned ephemeral processes, lab surfaces, immutable raw evidence."""
from __future__ import annotations
from gates import demand
import datetime,hashlib,json,os,psutil,signal,socket,statistics,subprocess,sys,threading,time,urllib.request,urllib.error
from pathlib import Path

S=Path("/mnt/hdd/lab/sessions/gnostral-q010-cpu-moe-hotpath-retest-20261010")
L=Path("/mnt/hdd/lab/sessions/gnostral-q009-llamacpp-cuda-20261010")
A=Path("/mnt/hdd/lab/sessions/gnostral-q009-strata-ab-ssd-20261009")
MODEL=Path("/mnt/models/active/gnostral/Qwen3-30B-A3B-Q2_K.gguf")
OLLAMA_MODEL_BASE=Path("/mnt/sandbox/sessions/gnostral-q009-ollama-compare-20261010/ollama/models")
LLAMA=Path("/mnt/sandbox/sessions/gnostral-q009-llamacpp-cuda-20261010/build/llama-cuda/bin/llama-server")
STRATA=Path("/home/tension_atoi/Projects/gnostral.rs/vendor/gnostral-strata-worktree/target-strata-candidate-release/release/mistralrs")
PROFILE=Path("/home/tension_atoi/Projects/gnostral.rs/vendor/gnostral-strata-worktree/harness/profiles/qwen3-30b-a3b-q2k-deterministic-4-per-layer.json")
PORTS={"baseline":18942,"candidate":18942}
EXPECTED={
 "model":"db3ce897ccc9e7d9dbf17fe083cae7880a2092aa473b45eba8b77715aa9ca170",
 "strata":"67743dbbbb04eca14498f8c79f57922e703111bbf80df7cdfc3918607a85ffb9",
 "llama":"8291521b7077047787c1617257c2f790bf377c1851b824f09064cd4195ab6be2",
 "profile":"d586ef8092fa53b72086934f007a424bfb0a4a26d8783bc3a7439ffe5a9915d0",
 "ollama_import":"0c9ccb6f83225157b66006fb3d86164b2df173582edbe701fd82cb1e95d6dae0"}
PROMPTS={
 "128":"In exactly two short sentences, explain one concrete benefit and one concrete drawback of keeping MoE expert weights in CPU RAM instead of GPU VRAM. Do not include headers or a list.",
 "512":"Provide exactly six numbered, concise checks for benchmarking a local mixture-of-experts model on a GPU with 8 GiB VRAM. Each check must contain one practical criterion and one way to verify it. Cover these six topics in this order: GGUF identity hash, CPU versus GPU layer placement, GPU and host memory peaks, first-text latency, steady decode tokens per second, and generated-answer correctness. Stop immediately after item six."
}
def raw_prompt(s):
 return "<|im_start|>user\n"+s+" /no_think<|im_end|>\n<|im_start|>assistant\n"
def hash_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def now():
 return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".partial")
 with tmp.open("w",encoding="utf-8") as f:
  json.dump(obj,f,ensure_ascii=False,indent=2)
  f.write("\n")
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def gpu():
 x=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=10)
 a=list(map(int,x.strip().split(", ")))
 return {"used_mib":a[0],"free_mib":a[1],"utilization_pct":a[2]}
def host_mib():
 with open("/proc/meminfo") as f:
  for line in f:
   if line.startswith("MemAvailable:"):return int(line.split()[1])/1024
 raise RuntimeError("MemAvailable not found")
def proc_rss_mib(proc):
 try:
  p=psutil.Process(proc.pid)
  children=[p]+p.children(recursive=True)
  return sum(x.memory_info().rss for x in children if x.is_running())/1024**2
 except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
  return 0.
def descendants(proc):
 try:return [x.pid for x in psutil.Process(proc.pid).children(recursive=True)]
 except psutil.NoSuchProcess:return []
class Sampler:
 def __init__(self,proc):
  self.proc=proc;self.done=threading.Event();self.samples=[]
  self.thread=threading.Thread(target=self.loop,daemon=True)
 def loop(self):
  while not self.done.is_set():
   try:
    x=gpu()
    self.samples.append({"t":time.time(),"used_gpu_mib":x["used_mib"],"free_gpu_mib":x["free_mib"],
      "gpu_util_pct":x["utilization_pct"],"host_available_mib":host_mib(),"owned_rss_mib":proc_rss_mib(self.proc)})
   except Exception as exc:self.samples.append({"error":repr(exc)})
   self.done.wait(.8)
 def start(self):self.thread.start()
 def end(self):self.done.set();self.thread.join(timeout=5)
 def summary(self):
  x=[v for v in self.samples if "error" not in v]
  return {"samples":len(x),"interval_target_s":.8,
   "peak_total_gpu_used_mib":max([v["used_gpu_mib"] for v in x],default=None),
   "peak_owned_process_tree_rss_mib":max([v["owned_rss_mib"] for v in x],default=None),
   "min_host_available_mib":min([v["host_available_mib"] for v in x],default=None),
   "peak_gpu_util_pct":max([v["gpu_util_pct"] for v in x],default=None)}
def post(url,obj,timeout=300):
 req=urllib.request.Request(url,data=json.dumps(obj).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def stream_ttft(url,obj,kind,timeout=300):
 payload=dict(obj);payload["stream"]=True
 req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
 start=time.perf_counter();first=None;num_chunks=0;max_chars=0
 with urllib.request.urlopen(req,timeout=timeout) as res:
  for line in res:
   if not line:continue
   if kind=="ollama":
    line=line.strip()
    if not line:continue
    x=json.loads(line);text=x.get("response") or ""
    if x.get("done"):break
   else:
    if not line.startswith(b"data: "):continue
    raw=line[6:].strip()
    if raw==b"[DONE]":break
    x=json.loads(raw)
    if kind=="llama": text=x.get("content") or ""
    else:
     ch=x.get("choices",[])
     firstchoice=ch[0] if ch else {}
     text=firstchoice.get("text") or firstchoice.get("delta",{}).get("content") or ""
   if text:
    num_chunks+=1;max_chars+=len(text)
    if first is None:first=round((time.perf_counter()-start)*1000,3)
   if time.perf_counter()-start>timeout:raise TimeoutError("TTFT request exceeded deadline")
 return {"first_text_ms":first,"stream_nonempty_chunks":num_chunks,"stream_text_chars":max_chars,
         "total_stream_wall_ms":round((time.perf_counter()-start)*1000,3),
         "provenance":"separate streamed request; first nonempty text, not nonstreamed measurement"}
def await_ready(proc,port,endpoint,timeout=150):
 started=time.perf_counter()
 while time.perf_counter()-started<timeout:
  if proc.poll() is not None:raise RuntimeError(f"server exited prematurely: {proc.returncode}")
  try:
   with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}",timeout=3) as r:
    if r.status==200:return round(time.perf_counter()-started,3)
  except (OSError,urllib.error.URLError,TimeoutError):pass
  time.sleep(.35)
 raise TimeoutError("readiness timeout")
def owned_stop(proc):
 before=descendants(proc)
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  proc.wait(timeout=10)
 for pid in before:
  try:
   p=psutil.Process(pid)
   if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
    p.terminate();p.wait(timeout=3)
  except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.TimeoutExpired):pass
 return proc.poll() is not None
def request_payload(engine,prompt,tokens):
 if engine=="ollama":
  return {"model":"q009-qwen3-q2k:smoke","prompt":prompt,"raw":True,"stream":False,"keep_alive":"10m",
       "options":{"num_ctx":1024,"num_predict":tokens,"temperature":0,"seed":42,"num_thread":8}}
 if engine=="llama":
  return {"prompt":prompt,"n_predict":tokens,"temperature":0,"seed":42,"top_p":1,"cache_prompt":False,"stream":False}
 return {"model":"default","prompt":prompt,"max_tokens":tokens,"temperature":0,"top_p":1,"stream":False}
def extract(engine,r):
 if engine=="ollama":
  tokens=r.get("eval_count")
  duration=r.get("eval_duration")
  return {"output":r.get("response"),"tokens":tokens,
    "tps_engine":tokens/(duration/1e9) if duration and tokens else None,
    "prompt_tokens_engine":r.get("prompt_eval_count"),
    "prefill_ms_engine":r.get("prompt_eval_duration",0)/1e6,
    "load_ms_engine":r.get("load_duration",0)/1e6,
    "source":"ollama /api/generate eval_count/eval_duration"}
 if engine=="llama":
  t=r.get("timings",{})
  return {"output":r.get("content"),"tokens":t.get("predicted_n"),
    "tps_engine":t.get("predicted_per_second"),"prompt_tokens_engine":t.get("prompt_n"),
    "prefill_ms_engine":t.get("prompt_ms"),"load_ms_engine":None,
    "source":"llama-server /completion timings"}
 u=r.get("usage",{})
 return {"output":r.get("choices",[{}])[0].get("text"),"tokens":u.get("completion_tokens"),
   "tps_engine":u.get("avg_compl_tok_per_sec"),"prompt_tokens_engine":u.get("prompt_tokens"),
   "prefill_ms_engine":u.get("total_prompt_time_sec",0)*1000,
   "load_ms_engine":None,"source":"mistralrs /v1/completions usage"}

def run(engine,repeat):
 demand(engine in PORTS and repeat in (1,2,3), "gate at q010_hotpath_ab_retest_20261010.py:170")
 port=PORTS[engine]
 with socket.socket() as sock:
  if sock.connect_ex(("127.0.0.1",port))==0:raise RuntimeError(f"port {port} already listening")
 before=gpu();ram=host_mib()
 r={"engine":engine,"repeat":repeat,"date_utc":now(),"status":"NOT_QUALIFIED",
    "before":{"gpu":before,"ram_available_mib":ram},"ctx_requested":1024,
    "dtype_rule":"Strata F32 + CPU KV F32; Ollama/llama use native model-specific precision",
    "samples":{}}
 if before["free_mib"]<1536 or ram<16384:
  r["status"]="RESOURCE_RESERVE_BLOCK";return r
 env=os.environ.copy()
 env.pop("GNOSTRAL_EXPERT_PROFILE",None);env.pop("GNOSTRAL_EXPERT_CACHE_BYTES",None)
 if engine=="ollama":
  env.update({"OLLAMA_HOST":"127.0.0.1:18945","OLLAMA_NO_CLOUD":"1",
   "OLLAMA_MODELS":str(OLLAMA_MODEL_BASE),"OLLAMA_NUM_PARALLEL":"1",
   "OLLAMA_MAX_LOADED_MODELS":"1","OLLAMA_MAX_QUEUE":"1","OLLAMA_KEEP_ALIVE":"10m"})
  argv=["/usr/local/bin/ollama","serve"];endpoint="/api/version"
 elif engine=="llama":
  argv=[str(LLAMA),"-m",str(MODEL),"-ngl","auto","-c","1024","-t","8",
    "--parallel","1","--host","127.0.0.1","--port","18946","--no-webui","--seed","42",
    "--cpu-moe","--moe-cache-mib","0"]
  endpoint="/health"
 else:
  prot=json.loads((A/"protocol-ssd-three-pairs.json").read_text())
  lane=prot["lanes"]["candidate"];argv=list(lane["command"]);env.update(lane["env"])
  if engine=="candidate":
   argv[0]=str(Path("/mnt/sandbox/sessions/gnostral-q010-cpu-moe-hotpath-20261010/target/release/mistralrs"))
  for flag in ("--pa-context-len","--max-model-len"):
   argv[argv.index(flag)+1]="1024"
  demand(argv[argv.index("-m")+1]==str(MODEL.parent), "gate at q010_hotpath_ab_retest_20261010.py:200")
  endpoint="/v1/models"
 log=S/f"{repeat:02d}-{engine}.server.log"
 proc=None;mon=None;lf=None
 try:
  lf=log.open("wb")
  proc=subprocess.Popen(argv,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
  mon=Sampler(proc);mon.start()
  r["server_start_cmd_redacted"]=argv
  r["ready_s"]=await_ready(proc,port,endpoint)
  r["warmup_result"]=extract(engine,post(f"http://127.0.0.1:{port}"+("/api/generate" if engine=="ollama" else "/completion" if engine=="llama" else "/v1/completions"),
       request_payload(engine,"2+3=",3),180))["output"]
  for budget in (128,512):
   p=raw_prompt(PROMPTS[str(budget)])
   payload=request_payload(engine,p,budget)
   url=f"http://127.0.0.1:{port}"+("/api/generate" if engine=="ollama" else "/completion" if engine=="llama" else "/v1/completions")
   begin=time.perf_counter()
   result=post(url,payload,300)
   parsed=extract(engine,result)
   parsed["nonstream_wall_s"]=round(time.perf_counter()-begin,3)
   if not isinstance(parsed["output"],str) or not parsed["output"]:raise RuntimeError(f"empty output {budget}")
   if not isinstance(parsed["tokens"],int) or not 1<=parsed["tokens"]<=budget:raise RuntimeError(f"invalid tokens {parsed['tokens']}")
   if not isinstance(parsed["tps_engine"],(int,float)) or parsed["tps_engine"]<=0:raise RuntimeError("invalid token rate")
   raw_output=S/f"{repeat:02d}-{engine}-{budget}-response.txt"
   raw_output.write_text(parsed.pop("output"),encoding="utf-8")
   parsed["output_path"]=str(raw_output);parsed["output_sha256"]=hash_file(raw_output)
   if budget==128:
    try:
     parsed["separate_request_stream"]=stream_ttft(url,request_payload(engine,p,128),engine,300)
    except Exception as exc:parsed["separate_request_stream"]={"status":"NOT_AVAILABLE","reason":repr(exc)}
   r["samples"][str(budget)]=parsed
   atomic_json(S/f"{repeat:02d}-{engine}.json",r)
   print("SAMPLE",repeat,engine,budget,"TOKENS",parsed["tokens"],"TPS",round(parsed["tps_engine"],3),
     "WALL",parsed["nonstream_wall_s"],flush=True)
   if host_mib()<8000 or gpu()["free_mib"]<300:raise RuntimeError("runtime resource emergency")
  r["status"]="PASS_SAMPLES"
 except Exception as exc:
  r["status"]="TRIAL_FAIL";r["error"]=repr(exc)
 finally:
  if mon is not None:mon.end();r["telemetry"]=mon.summary()
  if proc is not None:r["server_reaped"]=owned_stop(proc)
  else:r["server_reaped"]=True
  if lf is not None:lf.close()
  try:r["after"]=gpu();r["reclaim_delta_mib"]=r["after"]["used_mib"]-before["used_mib"]
  except Exception as exc:r["postflight_error"]=repr(exc)
  if r.get("server_reaped") is False or abs(r.get("reclaim_delta_mib",99999))>128:
   r["status"]="TRIAL_FAIL";r["reclaim_failure"]=True
  atomic_json(S/f"{repeat:02d}-{engine}.json",r)
 return r

def preflight():
 demand(str(S)==os.environ.get("LAB_SESSION"), "gate at q010_hotpath_ab_retest_20261010.py:251")
 demand(str(Path(os.environ["LAB_SCRATCH"])).startswith("/mnt/sandbox/sessions/"), "gate at q010_hotpath_ab_retest_20261010.py:252")
 demand(hash_file(MODEL)==EXPECTED["model"], "gate at q010_hotpath_ab_retest_20261010.py:253")
 demand(hash_file(STRATA)==EXPECTED["strata"], "gate at q010_hotpath_ab_retest_20261010.py:254")
 candidate=Path("/mnt/sandbox/sessions/gnostral-q010-cpu-moe-hotpath-20261010/target/release/mistralrs")
 demand(candidate.is_file(), "gate at q010_hotpath_ab_retest_20261010.py:256")
 candidate_sha=hash_file(candidate)
 demand(candidate_sha!=EXPECTED["strata"], "gate at q010_hotpath_ab_retest_20261010.py:258")
 (S/"candidate-hash-live.txt").write_text(candidate_sha+"\n")
 demand(hash_file(LLAMA)==EXPECTED["llama"], "gate at q010_hotpath_ab_retest_20261010.py:260")
 demand(hash_file(PROFILE)==EXPECTED["profile"], "gate at q010_hotpath_ab_retest_20261010.py:261")
 converted=OLLAMA_MODEL_BASE/"blobs/sha256-0c9ccb6f83225157b66006fb3d86164b2df173582edbe701fd82cb1e95d6dae0"
 # Ollama is not a participant in this matched-VRAM two-engine study.
 demand(not (S/"summary.json").exists(),"refuse to overwrite prior benchmark")
 from subprocess import check_output
 frozen={"schema":"q010-strata-q2kq3k-cpu-borrowed-ab-v1","source_sha256":EXPECTED,
  "ctx_requested":1024,"seed":42,"threads":8,"repeats":3,
  "repetition_order":["baseline","candidate","candidate","baseline","baseline","candidate"],
  "lengths":[128,512],"prompts":PROMPTS,
  "hypothesis":"Borrowed CPU Q2K/Q3K GGUF projections eliminate repeated packed-weight cloning and improve end-to-end MoE decode without shifting VRAM or host RSS",
  "predeclared_success_gate":{
     "candidate_median_tps_ratio_minimum":1.10,
     "same_or_equivalent_full_text_outputs":True,
     "all_six_reclaim_and_driver_checks_pass":True,
     "max_allowed_increase_gpu_peak_mib":128,
     "max_allowed_increase_host_rss_mib":512,
     "inference_quality_cannot_be_determined_from_tps":True
  },
  "interpretation":"n=3 descriptive A/B; speedup never generalized, no production promotion",
  "candidate_optimization":"CPU Q2K Q3K borrowed expert projections, no per-route packed weight copy",
  "comparison_target":"same original Strata binary vs only Q2K/Q3K CPU projection borrowed implementation, all other configs identical",
  "TTFT":"first nonempty text in separate 128-token streamed request; NOT the same request as nonstream throughput",
  "model_comparability":"same source GGUF and tokenizer, same Strata runtime configuration with one isolated source patch",
  "deployment":"lab only; no permanent service; run serially; safe preflight/reclaim",
  "memory":"sample whole-card used MiB and owned process tree RSS every .8s",
  "quality":"raw text retained; early EOG and task adherence evaluated separately",
  "biases":["3 repetitions insufficient for population inference","candidate binary revision fixed at benchmark start"]}
 atomic_json(S/"frozen-protocol.json",frozen)
 print("PREFLIGHT_PASS",now(),"model",EXPECTED["model"],"protocol",hash_file(S/"frozen-protocol.json"),"candidate_borrowed_Q2K_Q3K",flush=True)
 return frozen

def main():
 protocol=preflight();results=[]
 try:
  for i,engine in enumerate(protocol["repetition_order"]):
   repetition=1+sum(x["engine"]==engine for x in results)
   if repetition>3:raise AssertionError("invalid repetition order")
   print("BEGIN",i+1,len(protocol["repetition_order"]),engine,repetition,now(),flush=True)
   r=run(engine,repetition)
   results.append(r)
   if r["status"]!="PASS_SAMPLES":
    print("STOP_FAIL_CLOSED",engine,repetition,r.get("error",r["status"]),flush=True)
    break
 finally:
  counts={}
  for x in results:counts[x["engine"]]=counts.get(x["engine"],0)+1
  complete=len(results)==6 and all(x["status"]=="PASS_SAMPLES" for x in results)
  summary={"status":"ALL_SIX_TRIALS_PASS" if complete else "NOT_QUALIFIED",
   "completed":len(results),"by_engine":counts,"evidence_paths":[str(S/f"{x['repeat']:02d}-{x['engine']}.json") for x in results],
   "end_utc":now(),"repetition_protocol":"fresh server per repetition; both token ceilings per process; balanced blocks, direct 2-engine VRAM comparison",
   "speedup_claim":None}
  if complete:
   summary["median_decode_tps"]={engine:{str(n):statistics.median(x["samples"][str(n)]["tps_engine"]
    for x in results if x["engine"]==engine) for n in (128,512)} for engine in PORTS}
   summary["median_gpu_peak_total_mib"]={engine:statistics.median(x["telemetry"]["peak_total_gpu_used_mib"]
    for x in results if x["engine"]==engine) for engine in PORTS}
   summary["median_owned_rss_peak_mib"]={engine:statistics.median(x["telemetry"]["peak_owned_process_tree_rss_mib"]
    for x in results if x["engine"]==engine) for engine in PORTS}
  atomic_json(S/"summary.json",summary)
  print("COMPLETE",summary["status"],"by_engine",counts,"UTC",summary["end_utc"],flush=True)
 return 0 if complete else 7

if __name__=="__main__":
 sys.exit(main())
