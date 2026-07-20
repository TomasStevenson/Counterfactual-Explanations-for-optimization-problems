#!/bin/bash
# Envia los 18 jobs restantes (el piloto i14_fw se envia aparte).
cd ~/wcep
for f in job_*.sh; do
  [ "$f" = "job_i14_fw.sh" ] && continue
  jid=$(sbatch "$f" | awk '{print $4}')
  echo "$jid  $f" | tee -a submitted_jobs.txt
done
