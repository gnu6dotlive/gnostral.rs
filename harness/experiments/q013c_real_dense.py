#!/usr/bin/env python3
"""Q013-C pilot: exact SHA-pinned, one-shot dense inference in the bounded GNU6 scope.

This is an evidence-producing experimental harness, NOT an engine adapter deployed
for arbitrary users. It never accepts an arbitrary executable, model, port or URL.
"""
from __future__ import annotations
from gates import demand
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
Q012 = ROOT / "evidence/runs/gnostral-q012-native-three-families-20261010.json"
ENGINE = Path(os.environ.get("GNOSTRAL_Q013C_ENGINE", "/mnt/sandbox/sessions/gnostral-q012-native-capabilities-20261010/target/release/mistralrs"))
MODEL = Path(os.environ.get("GNOSTRAL_Q013C_DENSE_MODEL", str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a")))
WEIGHT = MODEL / "model.safetensors"
BINARY_SHA = "64eec222ccefff4cd4c0067bd83348f1d22eebda27d0d857e352ee090cf19723"
MODEL_SHA = "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"
Q012_SHA = "48241e35df171941e5d80a8f3832a55642de43d96abad04c0a000208e1cd3164"
PORT = 18949
MIN_RAM_MIB = 18000
MIN_VRAM_MIB = 5200
MAX_TOTAL_S = 125
MAX_RESPONSE_BYTES = 65536
PROBE = {"model":"default","messages":[{"role":"user","content":"Answer with only the word GNOSTRAL, in uppercase."}],
         "max_tokens":35,"temperature":0,"stream":False}

def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()

def number(path: Path) -> int:
    return int(path.read_text().strip())

def boundary() -> dict:
    path=next((line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
               if line.startswith("0::")),None)
    if not path or "/gnu6-lab.slice/" not in path or not path.endswith(".scope"):
        raise RuntimeError("DENY_UNBOUNDED_CGROUP")
    cg=Path("/sys/fs/cgroup")/path.lstrip("/")
    parent=cg.parent
    demand(parent.name=="gnu6-lab.slice","DENY_CGROUP_PARENT")
    vals={x:number(cg/x) for x in ("memory.max","memory.swap.max","memory.oom.group","memory.current","memory.swap.current")}
    vals.update({"parent_"+x:number(parent/x) for x in ("memory.high","memory.max","memory.swap.max")})
    if (vals["memory.max"]>21*1024**3 or vals["memory.swap.max"]>2*1024**3 or vals["memory.oom.group"]!=1 or
        vals["parent_memory.high"]>18*1024**3 or vals["parent_memory.max"]>22*1024**3 or vals["parent_memory.swap.max"]>2*1024**3):
        raise RuntimeError("DENY_CGROUP_BUDGET")
    vals["cgroup"]=path
    return vals

def gpu() -> dict:
    out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True,timeout=10)
    rows=out.strip().splitlines()
    if len(rows)!=1:raise RuntimeError("DENY_AMBIGUOUS_GPU")
    used,free,load=(int(x.strip()) for x in rows[0].split(","))
    return {"used_mib":used,"free_mib":free,"utilization_pct":load}

def ram_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):return int(line.split()[1])//1024
    raise RuntimeError("RAM_UNKNOWN")

def compositor() -> None:
    subprocess.run(["systemctl","--user","is-active","--quiet",
                    "wayland-wm@hyprland.desktop.service"],check=True,timeout=10)

def no_other_engine() -> None:
    result=subprocess.run(["pgrep","-x","mistralrs"],capture_output=True,timeout=5)
    if result.returncode==0:raise RuntimeError("DENY_ENGINE_ALREADY_RUNNING")
    if result.returncode!=1:raise RuntimeError("ENGINE_PROCESS_CHECK_FAILED")

def port_free() -> None:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1",PORT))==0:raise RuntimeError("DENY_PORT_IN_USE")

def http_json(endpoint: str, payload: dict|None=None, timeout: float=3) -> dict:
    addr=f"http://127.0.0.1:{PORT}{endpoint}"
    data=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request(addr,data=data,headers={"Content-Type":"application/json"},method="POST" if data else "GET")
    with urllib.request.urlopen(request,timeout=timeout) as response:
        body=response.read(MAX_RESPONSE_BYTES+1)
        if len(body)>MAX_RESPONSE_BYTES:raise RuntimeError("DENY_OVERSIZE_HTTP_RESPONSE")
    return json.loads(body)

def persist_new(path:Path,record:dict) -> None:
    with path.open("x") as out:json.dump(record,out,sort_keys=True,indent=2);out.write("\n")

def prepare_snapshot(src:Path,dst:Path) -> None:
    with src.open("rb") as inp, dst.open("xb") as out:
        shutil.copyfileobj(inp,out,length=4*1024*1024)
        out.flush();os.fsync(out.fileno())
    os.chmod(dst,0o500)
    if sha(dst)!=BINARY_SHA:raise RuntimeError("DENY_SNAPSHOT_SHA_MISMATCH")

def preflight() -> dict:
    compositor()
    cg=boundary()
    no_other_engine()
    port_free()
    q=Q012.read_bytes()
    if hashlib.sha256(q).hexdigest()!=Q012_SHA:raise RuntimeError("DENY_Q012_RECEIPT_CHANGED")
    published=json.loads(q)
    d=published.get("profiles",{}).get("dense",{})
    if published.get("verdict")!="THREE_FAMILIES_SAME_BINARY_PASS" or published.get("native_binary_sha256")!=BINARY_SHA or d.get("model_weight_sha256")!=MODEL_SHA or d.get("status")!="PASS_FUNCTIONAL":
        raise RuntimeError("DENY_Q012_QUALIFICATION_MISMATCH")
    before=gpu()
    available=ram_available()
    if before["free_mib"]<MIN_VRAM_MIB or available<MIN_RAM_MIB:raise RuntimeError("DENY_INSUFFICIENT_HEADROOM")
    if sha(ENGINE)!=BINARY_SHA:raise RuntimeError("DENY_EXECUTABLE_IDENTITY")
    if sha(WEIGHT)!=MODEL_SHA:raise RuntimeError("DENY_MODEL_WEIGHT_IDENTITY")
    return {"gpu_before":before,"ram_available_mib":available,"boundary":cg,
            "binary_sha256":BINARY_SHA,"model_weight_sha256":MODEL_SHA,
            "q012_evidence_sha256":Q012_SHA}

def stop_owned(proc:subprocess.Popen|None) -> bool:
    if proc is None:return True
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=7)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=7)
    return proc.poll() is not None

def run(session:Path) -> dict:
    if session.exists():raise RuntimeError("DENY_EVIDENCE_DIRECTORY_EXISTS")
    session.mkdir(parents=True,mode=0o700)
    os.chmod(session,0o700)
    record={"schema":"gnostral.q013c.real-dense-pilot.v1","started_utc":utc(),
            "status":"NOT_QUALIFIED","scope":"single isolated ephemeral dense Rust/CUDA server only",
            "nonclaims":["MoE/embedding adapter","persistent model serving","hot-swap","production",
                         "authenticated independent semantic witness","container security against same-user adversary"]}
    proc=None
    log=None
    temp=None
    started=time.monotonic()
    try:
        check=preflight()
        record["preflight"]=check
        persist_new(session/"preflight.json",check)
        temp=Path(tempfile.mkdtemp(prefix="engine-",dir=session))
        os.chmod(temp,0o700)
        binary=temp/"mistralrs"
        prepare_snapshot(ENGINE,binary)
        record["snapshot_sha256"]=sha(binary)
        logfile=session/"engine.log"
        log=logfile.open("xb")
        args=[str(binary),"serve","--host","127.0.0.1","--port",str(PORT),
              "text","--model-id",str(MODEL),"--max-seq-len","1024",
              "--paged-attn","off","--token-source","none","--max-batch-size","1"]
        env=os.environ.copy()
        env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",
                   HF_DATASETS_OFFLINE="1",RAYON_NUM_THREADS="8")
        proc=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              env=env,start_new_session=True)
        record["lifecycle"]=["PREFLIGHT","SNAPSHOT_VERIFIED","STARTED"]
        ready_deadline=min(started+90,time.monotonic()+85)
        while time.monotonic()<ready_deadline:
            if proc.poll() is not None:raise RuntimeError(f"MODEL_EXIT_BEFORE_READY:{proc.returncode}")
            compositor()
            try:
                model_info=http_json("/v1/models",timeout=2)
                if isinstance(model_info.get("data"),list) and model_info["data"]:
                    record["ready_after_s"]=round(time.monotonic()-started,3)
                    break
            except (urllib.error.URLError,TimeoutError,ValueError,ConnectionError):
                time.sleep(0.4)
        else:raise RuntimeError("MODEL_READY_TIMEOUT")
        record["lifecycle"].append("HTTP_READY")
        if time.monotonic()-started>MAX_TOTAL_S-65:raise RuntimeError("DEADLINE_NEAR")
        response=http_json("/v1/chat/completions",PROBE,timeout=55)
        answer=response["choices"][0]["message"]["content"]
        if not isinstance(answer,str) or not answer.strip():raise RuntimeError("EMPTY_SEMANTIC_OUTPUT")
        persist_new(session/"response-private.json",response)
        record["raw_output_sha256"]=hashlib.sha256(answer.encode()).hexdigest()
        record["response_file_sha256"]=sha(session/"response-private.json")
        record["observed_answer"]=answer[:200]
        record["usage"]=response.get("usage",{})
        record["lifecycle"].append("SEMANTIC_RESPONSE_RETAINED")
    except Exception as exc:
        record["failure"]=repr(exc)
    finally:
        try:
            record["process_reaped"]=stop_owned(proc)
        except Exception as exc:
            record["process_reaped"]=False
            record["stop_failure"]=repr(exc)
        if log:log.close()
        if temp:
            shutil.rmtree(temp,ignore_errors=True)
        record["elapsed_s"]=round(time.monotonic()-started,3)
        try:
            record["gpu_after"]=gpu()
            if "preflight" in record:
                delta=record["gpu_after"]["used_mib"]-record["preflight"]["gpu_before"]["used_mib"]
                record["gpu_ambient_delta_mib"]=delta
                record["reclaim_within_128_mib"]=abs(delta)<=128
        except Exception as exc:record["telemetry_failure"]=repr(exc)
        record["ended_utc"]=utc()
        record["status"]="RUN_COMPLETED_PENDING_INDEPENDENT_AUDIT" if (
            record.get("process_reaped") and record.get("reclaim_within_128_mib") and
            "SEMANTIC_RESPONSE_RETAINED" in record.get("lifecycle",[])
        ) else "NOT_QUALIFIED"
        persist_new(session/"session.json",record)
    return record

def main() -> int:
    if len(sys.argv)!=2:
        print("USAGE: q013c_real_dense.py PRIVATE_NEW_SESSION_DIRECTORY",file=sys.stderr)
        return 64
    outcome=run(Path(sys.argv[1]))
    print("Q013C_REAL_DENSE",outcome["status"],"elapsed",outcome["elapsed_s"],
          "reclaim",outcome.get("gpu_ambient_delta_mib"),"error",outcome.get("failure",""),flush=True)
    return 0 if outcome["status"]=="RUN_COMPLETED_PENDING_INDEPENDENT_AUDIT" else 2

if __name__=="__main__":raise SystemExit(main())
