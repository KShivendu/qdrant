"""Log precise collection lifecycle: creation seen, first time fully indexed and green."""
import sys, time, json, urllib.request
out = open(sys.argv[1], "a"); url = "http://localhost:6333/collections/benchmark"
created = None; last = None
while True:
    try:
        d = json.load(urllib.request.urlopen(url, timeout=2))["result"]
        now = time.time()
        if created is None:
            created = now; out.write(json.dumps({"event": "created", "t": now}) + "\n"); out.flush()
        st = (d["status"], d.get("points_count"), d.get("indexed_vectors_count"), str(d.get("optimizer_status")))
        if st != last:
            out.write(json.dumps({"event": "status", "t": now, "status": st}) + "\n"); out.flush(); last = st
    except Exception:
        if created is not None:
            out.write(json.dumps({"event": "gone", "t": time.time()}) + "\n"); out.flush(); created = None; last = None
    time.sleep(0.2)
