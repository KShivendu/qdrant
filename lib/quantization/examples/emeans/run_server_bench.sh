# usage: run_server_bench.sh <order...>  e.g. old new new old old new
set -u
S=${S:?set S to a scratch dir with bin/, data/ and ref/vector-db-benchmark}
V=$S/ref/vector-db-benchmark
OUT=$S/server; mkdir -p $OUT
i=0
for v in "$@"; do
  i=$((i+1)); run=$OUT/run$i-$v; mkdir -p $run; rm -rf $S/qstorage; mkdir -p $S/qstorage
  if curl -sf localhost:6333/readyz >/dev/null; then echo "port busy before start" >&2; exit 1; fi
  (cd /home/user/qdrant && QDRANT__STORAGE__STORAGE_PATH=$S/qstorage QDRANT__STORAGE__SNAPSHOTS_PATH=$S/qstorage/snap \
     QDRANT__TELEMETRY_DISABLED=true QDRANT__LOG_LEVEL=INFO exec $S/bin/qdrant_$v > $run/qdrant.log 2>&1) &
  QPID=$!
  sleep 1
  until curl -sf localhost:6333/readyz >/dev/null; do sleep 0.5; done
  grep -q "Qdrant HTTP listening on 6333" $run/qdrant.log || { echo "run$i: server did not bind" >&2; exit 1; }
  python3 -I $S/scripts/poll_green.py $run/poll.jsonl & PPID2=$!
  rm -f $V/results/*.json
  (cd $V && .venv/bin/python run.py --engines "qdrant-pq-*" --datasets dbpedia-openai-100K-1536-angular > $run/vdb.log 2>&1)
  echo "run$i $v exit=$?" >&2
  cp $V/results/*.json $run/ 2>/dev/null
  kill $PPID2; kill $QPID; wait $QPID 2>/dev/null; pkill -f "bin/qdrant_"; while curl -sf localhost:6333/readyz >/dev/null; do sleep 0.5; done
done
echo SERVERDONE >&2
