#!/usr/bin/env bash
# 并行驱动 04_env_rollout.py 的全部分片。
#
# 用法：
#   bash recsim/gen/run_all_shards.sh 100 32          # 100 片，32 进程（50w 正式）
#   bash recsim/gen/run_all_shards.sh 4 4 _5000       # 4 片，4 进程，跑 5,000 人的计划表
#
# 本机 112 核，默认只用 32 个，留核给别人。
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=/home/fengyuebo/miniconda3/envs/py311/bin/python   # 见 PLAN §7：python3 是 3.8，不能用
NSHARD=${1:-100}
NPROC=${2:-32}
PLAN=${3:-}
LOGDIR=recsim/out/gen/shardlogs
mkdir -p "$LOGDIR"

PLANARG=()
[ -n "$PLAN" ] && PLANARG=(--plan "$PLAN")

echo "分片 $NSHARD 片 / 并行 $NPROC 进程 / 计划表后缀 '${PLAN:-（正式 50w）}'"
date

seq 0 $((NSHARD - 1)) | xargs -P "$NPROC" -I{} \
  bash -c "$PY recsim/gen/04_env_rollout.py --shard {} --nshard $NSHARD ${PLANARG[*]:-} \
           > $LOGDIR/shard${PLAN}_{}.log 2>&1 || echo 'FAILED shard {}'"

date
FAIL=$(grep -l Traceback "$LOGDIR"/shard${PLAN}_*.log 2>/dev/null | wc -l)
if [ "$FAIL" -gt 0 ]; then
  echo "❌ $FAIL 个分片报错，看 $LOGDIR"
  grep -h -A5 Traceback "$LOGDIR"/shard${PLAN}_*.log | head -40
  exit 1
fi
echo "✅ 全部 $NSHARD 片完成。下一步：$PY recsim/gen/05_merge.py ${PLANARG[*]:-}"
