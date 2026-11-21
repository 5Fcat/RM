#!/usr/bin/env bash
# 补充实验 B（claude/补充实验B_2026-11-21.md；只描述，不用 API）
#   B1 执行器使用情况：各运行最终最好设计 × 1 回合（平地、横梁、敏感性、v370 提示词、3 种新地形；223 个）
#   B3 跨地形交叉评估：5 种地形上 GA / LLM 变异 + ② 的最终最好设计（100 个）× 5 种地形 × 2 回合
#   B2 最优设计重评的补充：H6（confirm_beams）与 3 种新地形的最终最好设计 × 10 回合（132 个运行）
#   结果 mac_runs/supp_B/{actuator,cross,reeval}/；都可中断，重新启动从断点继续
# 启动：push_tools.bat → run_mac.bat 输入 supp_b_mac → wait_mac.bat
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
F="$RUNS/terrain_ext/formal"
OUT="$RUNS/supp_B"
mkdir -p "$OUT"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
while pgrep -f "wj_tools/terrain_ext_mac" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.reeval_best" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$OUT/tests.txt" 2>&1
rc=$?
grep -v -i warn "$OUT/tests.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/supp_B/tests.txt）"; exit 1; }

echo "=== B1 执行器使用情况 $(date '+%F %T')"
uv run --locked python -m hyblock.supp_b actuator --threads 12 --out "$OUT/actuator" --roots \
    "$RUNS/formal_flat" "$RUNS/formal_beams" "$RUNS/sensitivity/p14" "$RUNS/sensitivity/m15" "$RUNS/sensitivity/v370" \
    "$RUNS/sensitivity/v370_prompt370" "$F/ridged_s0.75" "$F/slope_5" "$F/beams_s1.1" 2>&1 | grep -v -i warn | tail -3
echo "=== B3 跨地形交叉评估 $(date '+%F %T')"
uv run --locked python -m hyblock.supp_b cross --threads 12 --out "$OUT/cross" --roots \
    "$RUNS/formal_flat" "$RUNS/formal_beams" "$F/ridged_s0.75" "$F/slope_5" "$F/beams_s1.1" 2>&1 | grep -v -i warn | tail -3
echo "=== B2 最优设计重评（补充） $(date '+%F %T')"
uv run --locked python -m hyblock.reeval_best run --threads 12 --out "$OUT/reeval" --roots \
    "$RUNS/confirm_beams" "$F/ridged_s0.75" "$F/slope_5" "$F/beams_s1.1" 2>&1 | grep -v -i warn | tail -3
echo "=== 全部完成 $(date '+%F %T')"
