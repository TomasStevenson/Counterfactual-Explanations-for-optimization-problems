#!/bin/bash
# Poll Leftraru until all WCEP jobs finish, then summarize and pull logs.
cd /c/Users/tomas/Desktop/wcep_leftraru
while true; do
  left=$(ssh -o ConnectTimeout=30 leftraru 'squeue --me -h -o "%j"' 2>/dev/null | grep -cE "^i(14|39|57)_")
  ts=$(date "+%H:%M")
  if [ "$left" = "0" ]; then
    echo "[$ts] no WCEP jobs left in queue - collecting results"
    break
  fi
  echo "[$ts] $left WCEP jobs still queued/running"
  sleep 900
done
ssh leftraru 'sacct -X --name=i14_fw,i14_A_dual,i14_A_padm,i14_b_dual,i14_b_padm,i39_fw,i39_A_dual_v1,i39_A_dual_v2,i39_A_padm_v1,i39_A_padm_v2,i39_b_dual_v1,i39_b_dual_v2,i39_b_padm_v1,i39_b_padm_v2,i57_fw,i57_A_dual,i57_A_padm,i57_b_dual,i57_b_padm --starttime=2026-07-16 --format=JobID,JobName%16,State,Elapsed,MaxRSS,ExitCode' 2>/dev/null
echo "=== pulling logs ==="
scp -q "leftraru:~/wcep/*.out" "leftraru:~/wcep/*.err" results/ 2>/dev/null
ls results/ | wc -l
echo "=== [TIMING] summary ==="
grep -H "^\[TIMING\]" results/*.out
