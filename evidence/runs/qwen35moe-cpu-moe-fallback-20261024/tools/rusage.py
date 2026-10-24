import json, resource, subprocess, sys, time
cmd = sys.argv[1:]
t0 = time.monotonic()
p = subprocess.run(cmd, capture_output=True, text=True)
wall = time.monotonic() - t0
r = resource.getrusage(resource.RUSAGE_CHILDREN)
metrics = next((l for l in p.stdout.splitlines() if l.startswith("GNOSTRAL_METRICS")), "none")
print(json.dumps({"rc": p.returncode, "wall_s": round(wall, 2), "user_s": round(r.ru_utime, 2), "sys_s": round(r.ru_stime, 2),
                  "minor_faults": r.ru_minflt, "major_faults": r.ru_majflt, "vol_ctx": r.ru_nvcsw, "invol_ctx": r.ru_nivcsw,
                  "max_rss_mb": round(r.ru_maxrss / 1024), "metrics": metrics[17:]}))
if p.returncode: sys.stderr.write(p.stderr[-2000:])
