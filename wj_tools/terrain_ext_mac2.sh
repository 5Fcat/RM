#!/usr/bin/env bash
# 地形扩展 第 2 步：正式实验（claude/地形扩展计划.md 第 3 节；冻结记录写好之后才启动）
#   地形 = wj_tools/terrain_ext_frozen.txt（每行一个地形名；冻结时由 Claude 写入并推送，之后不改）
#   种子 500–509；每个种子、每种地形：
#     B1  普通 GA（ga，预算 200）
#     B5  GA + 地形版 LLM 变异的前 5 代（ga_llmtyped_pre，预算 80 = 第 0–4 代；只给 B6 复制，不单独分析）
#     B6  GA + 地形版 LLM 变异 + ② LLM Ev（ga_llmtyped_sel_llmEv，复制 B5 第 0–4 代：evals、llm_mut、llm_cache）
#   其余设置与横梁正式实验（formal_beams）相同。按种子交错，中途停下也能拿到完整的种子。
#   结果 mac_runs/terrain_ext/formal/<地形>/；最后写 analysis/report.md（hyblock/terrain_ext_analysis.py formal）
# 启动：push_tools.bat → run_mac.bat 输入 terrain_ext_mac2 → wait_mac.bat。随时可中断，重新启动从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
FO="$RUNS/terrain_ext/formal"
mkdir -p "$FO"
FROZEN=$(sed '/^#/d;/^$/d' wj_tools/terrain_ext_frozen.txt 2>/dev/null | tr '\n' ' ')
[ -n "$FROZEN" ] || { echo "!!! wj_tools/terrain_ext_frozen.txt 是空的：先写冻结记录再启动"; exit 1; }
EV="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --sel_pool 24 --sel_mode rank --sel_votes 3"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)  地形：${FROZEN}"
while pgrep -f "wj_tools/week15_mac1" >/dev/null || pgrep -f "wj_tools/terrain_ext_mac1" >/dev/null || pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$FO/tests_formal.txt" 2>&1
rc=$?
grep -v -i warn "$FO/tests_formal.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/terrain_ext/formal/tests_formal.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY（copy_env_to_mac.bat）"; exit 1; }

run() {   # run <输出根目录> <名称> <种子> <预算> <参数…>
  local root="$1" name="$2" s="$3" budget="$4"; shift 4
  local out="$root/${name}_s${s}"
  [ -f "$out/DONE" ] && return 0
  echo "=== $(basename "$root")/${name} 种子 ${s} $(date '+%F %T')"
  uv run --locked python -m hyblock.search --seed "$s" --budget "$budget" --batch 16 --max_modules 8 --threads 12 \
      --llm_workers 8 "$@" --out "$out" 2>&1 | grep -v -i warn | tail -2
  [ -f "$out/DONE" ] || echo "!!! ${name} 种子 ${s} 没有 DONE（继续下一个；重新启动本任务会从断点继续）"
}

copy_llm_mut() {   # copy_llm_mut <源目录> <目标目录>：复制 llm_mut.csv 的第 0–4 代与 llm_cache（只在目标还没开始时）
  local src="$1" out="$2"
  [ -f "$out/evals.csv" ] && return 0
  mkdir -p "$out"
  [ -d "$out/llm_cache" ] || cp -R "$src/llm_cache" "$out/llm_cache"
  uv run --locked python - "$src/llm_mut.csv" "$out/llm_mut.csv" <<'PY'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    rows = [x for x in r if int(x["gen"]) < 5]
    fields = r.fieldnames
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"llm_mut.csv：复制第 0–4 代 {len(rows)} 行")
PY
}

for s in 500 501 502 503 504 505 506 507 508 509; do
  for t in $FROZEN; do
    root="$FO/${t}"
    mkdir -p "$root"
    run "$root" ga "$s" 200 --method ga --terrain "$t"
    run "$root" ga_llmtyped_pre "$s" 80 --method ga --terrain "$t" --mutation llm_typed
    [ -f "$root/ga_llmtyped_pre_s${s}/DONE" ] || continue
    if [ ! -f "$root/ga_llmtyped_sel_llmEv_s${s}/DONE" ]; then
      copy_llm_mut "$root/ga_llmtyped_pre_s${s}" "$root/ga_llmtyped_sel_llmEv_s${s}" || { echo "!!! 复制失败，跳过"; continue; }
    fi
    run "$root" ga_llmtyped_sel_llmEv "$s" 200 --method ga --terrain "$t" --mutation llm_typed --select llm $EV \
        --init_from "$root/ga_llmtyped_pre_s${s}/evals.csv"
  done
done
uv run --locked python -m hyblock.terrain_ext_analysis formal --root "$FO" --flat "$RUNS/formal_flat" \
    --beams "$RUNS/formal_beams" --out "$FO/analysis" | tail -20
echo "=== 全部完成 $(date '+%F %T')"
