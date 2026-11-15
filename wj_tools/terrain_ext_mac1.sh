#!/usr/bin/env bash
# 地形扩展 第 1 步（claude/地形扩展计划.md = 正式实验计划 变更记录第 22 条；11-13 写定，数据出来之前）：
#   1. 离线测试（含 tests/test_terrains.py）
#   2. 难度标定：calib_sc 的 120 个随机设计里平地能走的，在 16 个候选难度上各跑 MPC 2 回合（不跑任何 LLM）；
#      没有难度落在 20–30% 的地形自动插一级再测（最多一次）→ mac_runs/terrain_ext/calib/（summary.md、chosen.json）
#   3. GA 试点：每种选定的难度，普通 GA，开发种子 0–2，预算 200 → mac_runs/terrain_ext/pilot/<地形>/ga_s<种子>
#      不通过（后期中位数均值 < 3 × 第 0 代中位数均值，或 < 20）→ 在"退一级"上重新试点（最多一次）
#   4. 试点分析 → mac_runs/terrain_ext/pilot/analysis/report.md、final.txt（进入正式实验的地形）
#   正式实验（种子 500–509）不在这里：看完结果、在计划里写好冻结记录之后，另用 terrain_ext_mac2.sh。
#   不用 API。预计：标定约 10–14 小时，试点约 35–50 小时。
# 启动：H6（week15_mac1）跑完之后再推送、启动——run_mac.bat 一启动就 git pull，会换掉 H6 剩下运行的代码。
#       Windows 双击 push_tools.bat → run_mac.bat 输入 terrain_ext_mac1 → wait_mac.bat。下面"等 week15_mac1"的检查只是双重保险。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
TE="$RUNS/terrain_ext"
CAL="$TE/calib"
PIL="$TE/pilot"
mkdir -p "$CAL" "$PIL"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
while pgrep -f "wj_tools/week15_mac1" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.reeval_best" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
[ -f "$RUNS/calib_sc/flat.csv" ] || { echo "!!! 缺少 mac_runs/calib_sc/flat.csv，停止"; exit 1; }

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$TE/tests_terrain_ext.txt" 2>&1
rc=$?
grep -v -i warn "$TE/tests_terrain_ext.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/terrain_ext/tests_terrain_ext.txt）"; exit 1; }

echo "=== 难度标定 $(date '+%F %T')"
[ -f "$CAL/DONE" ] || uv run --locked python wj_tools/run_meta.py start "$CAL" --threads 12 --note "terrain_ext calibration"
uv run --locked python -m hyblock.calib_terrain --flat "$RUNS/calib_sc/flat.csv" --out "$CAL" --threads 12 --auto_insert \
  2>&1 | grep -v -i warn | grep -v "^\[calib\]"
[ -f "$CAL/chosen.txt" ] || { echo "!!! 没有 chosen.txt，停止"; exit 1; }
uv run --locked python wj_tools/run_meta.py done "$CAL"

pilot() {   # pilot <地形>：普通 GA，种子 0–2
  local t="$1" s out
  for s in 0 1 2; do
    out="$PIL/${t}/ga_s${s}"
    [ -f "$out/DONE" ] && continue
    echo "=== 试点 ${t} 种子 ${s} $(date '+%F %T')"
    uv run --locked python -m hyblock.search --method ga --terrain "$t" --seed "$s" --budget 200 --batch 16 \
        --max_modules 8 --threads 12 --out "$out" 2>&1 | grep -v -i warn | tail -2
    [ -f "$out/DONE" ] || echo "!!! 试点 ${t} 种子 ${s} 没有 DONE"
  done
}

CHOSEN=$(awk 'NF == 2 {printf "%s:%s ", $1, $2}' "$CAL/chosen.txt")
echo "=== 选定（难度:退一级）：${CHOSEN}"
for pair in $CHOSEN; do pilot "${pair%%:*}"; done
uv run --locked python -m hyblock.terrain_ext_analysis pilot --root "$PIL" --out "$PIL/analysis" | tail -12

FINAL=""
for pair in $CHOSEN; do
  t="${pair%%:*}"; fb="${pair##*:}"
  v=$(grep -o "\"${t}\": \"[a-z]*\"" "$PIL/analysis/pilot_verdict.json" | sed 's/.*: "//;s/"//')
  if [ "$v" = "pass" ]; then FINAL="$FINAL $t"; continue; fi
  echo "--- ${t} 试点不通过（${v}）→ 退一级 ${fb}"
  pilot "$fb"
  uv run --locked python -m hyblock.terrain_ext_analysis pilot --root "$PIL" --out "$PIL/analysis" >/dev/null
  v2=$(grep -o "\"${fb}\": \"[a-z]*\"" "$PIL/analysis/pilot_verdict.json" | sed 's/.*: "//;s/"//')
  if [ "$v2" = "pass" ]; then FINAL="$FINAL $fb"; else echo "--- ${fb} 也不通过 → 该地形不进入正式实验"; fi
done
echo "$FINAL" | tr ' ' '\n' | sed '/^$/d' > "$PIL/analysis/final.txt"
uv run --locked python -m hyblock.terrain_ext_analysis pilot --root "$PIL" --out "$PIL/analysis" | tail -14
echo "=== 进入正式实验的地形：$(tr '\n' ' ' < "$PIL/analysis/final.txt")"
echo "=== 全部完成 $(date '+%F %T')"
