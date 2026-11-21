#!/usr/bin/env bash
# 换模型消融（claude/正式实验计划.md 变更记录第 23 条；只描述）
#   斜坡 5°（slope_5）、种子 500–509。模型 = wj_tools/model_ablation_models.txt（每行"标签 模型 ID"）
#   每个种子、每个模型：
#     B5′ 前缀：GA + 地形版 LLM 变异的前 5 代（ga_llmtyped_pre_<标签>，预算 80），LLM = 该模型
#     B6′：GA + 地形版 LLM 变异 + ② LLM Ev（ga_llmtyped_sel_llmEv_<标签>，复制 B5′ 第 0–4 代），变异与 ② 都用该模型
#   B1 与 Haiku 的 B6 用地形扩展正式实验的结果（mac_runs/terrain_ext/formal/slope_5/），不重跑。
#   其余设置与地形扩展正式实验（terrain_ext_mac2.sh）完全相同。按种子交错，中途停下也能拿到完整的种子。
#   结果 mac_runs/model_ablation/slope_5/；最后写 mac_runs/model_ablation/analysis/report.md
#   分批（11-22）：wj_tools/model_ablation_batch.txt 的 SEEDS（这一批的种子）与 MAX_USD（每个模型的费用估算上限，
#   hyblock/ablation_cost.py）。到上限就不再开始新的运行、正常结束；10 个种子都跑完之前不做比较分析。
# 启动：push_tools.bat → run_mac.bat 输入 model_ablation_mac → wait_mac.bat。随时可中断，重新启动从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
T=slope_5
BASE="$RUNS/terrain_ext/formal/$T"
MA="$RUNS/model_ablation"
ROOT="$MA/$T"
mkdir -p "$ROOT"
EV="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --sel_pool 24 --sel_mode rank --sel_votes 3"
MODELS=$(sed '/^#/d;/^[[:space:]]*$/d' wj_tools/model_ablation_models.txt 2>/dev/null)
[ -n "$MODELS" ] || { echo "!!! wj_tools/model_ablation_models.txt 是空的"; exit 1; }
TAGS=$(echo "$MODELS" | awk '{print $1}' | tr '\n' ' ')
ALL_SEEDS="500 501 502 503 504 505 506 507 508 509"
SEEDS="$ALL_SEEDS"; MAX_USD=""
if [ -f wj_tools/model_ablation_batch.txt ]; then
  v=$(sed -n 's/^SEEDS=//p' wj_tools/model_ablation_batch.txt | tail -1); [ -n "$v" ] && SEEDS="$v"
  MAX_USD=$(sed -n 's/^MAX_USD=//p' wj_tools/model_ablation_batch.txt | tail -1)
fi
for s in $SEEDS; do
  case " $ALL_SEEDS " in *" $s "*) ;; *) echo "!!! SEEDS 里的 ${s} 不在 500–509，停止"; exit 1;; esac
done
echo "=== start $(date '+%F %T')  $(git log --oneline -1)  模型：${TAGS} 这一批的种子：${SEEDS}  费用上限：${MAX_USD:-无}"
while pgrep -f "wj_tools/terrain_ext_mac" >/dev/null || pgrep -f "wj_tools/week15_mac1" >/dev/null || pgrep -f "wj_tools/supp_b_mac" >/dev/null || pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
for s in 500 501 502 503 504 505 506 507 508 509; do
  for g in ga ga_llmtyped_sel_llmEv; do
    [ -f "$BASE/${g}_s${s}/DONE" ] || { echo "!!! 缺地形扩展的 $T/${g}_s${s}（B1 / Haiku B6），停止"; exit 1; }
  done
done
echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$MA/tests.txt" 2>&1
rc=$?
grep -v -i warn "$MA/tests.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/model_ablation/tests.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY（copy_env_to_mac.bat）"; exit 1; }

budget_ok() {   # budget_ok <标签> <模型>：费用估算没到 MAX_USD 才返回 0（没设 MAX_USD 时总是 0）
  [ -n "$MAX_USD" ] || return 0
  uv run --locked python -m hyblock.ablation_cost --root "$ROOT" --tag "$1" --model "$2" --max "$MAX_USD" | tail -1
  local rc=${PIPESTATUS[0]}
  [ $rc -eq 0 ] && return 0
  [ $rc -eq 3 ] && { echo "=== ${1} 的费用估算到了上限 \$${MAX_USD}，不再开始新的运行"; return 1; }
  echo "!!! 费用估算出错（${rc}），停止"; exit 1
}

run() {   # run <名称> <种子> <预算> <参数…>；没有 DONE 就停止整个任务（多半是模型 ID 或 API 的问题；重新启动从断点继续）
  local name="$1" s="$2" budget="$3"; shift 3
  local out="$ROOT/${name}_s${s}"
  [ -f "$out/DONE" ] && return 0
  echo "=== ${name} 种子 ${s} $(date '+%F %T')"
  uv run --locked python -m hyblock.search --seed "$s" --budget "$budget" --batch 16 --max_modules 8 --threads 12 \
      --llm_workers 8 --method ga --terrain "$T" "$@" --out "$out" 2>&1 | grep -v -i warn | tail -2
  [ -f "$out/DONE" ] || { echo "!!! ${name} 种子 ${s} 没有 DONE，停止（日志交给 Claude；修好后重新启动会从断点继续）"; exit 1; }
}

copy_llm_mut() {   # 同 terrain_ext_mac2.sh：复制 llm_mut.csv 的第 0–4 代与 llm_cache（只在目标还没开始时）
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

STOPPED=""
for s in $SEEDS; do
  while read -r tag model <&3; do
    case " $STOPPED " in *" $tag "*) continue;; esac
    if [ ! -f "$ROOT/ga_llmtyped_sel_llmEv_${tag}_s${s}/DONE" ] && ! budget_ok "$tag" "$model"; then
      STOPPED="$STOPPED $tag"; continue
    fi
    run "ga_llmtyped_pre_${tag}" "$s" 80 --mutation llm_typed --model "$model"
    if [ ! -f "$ROOT/ga_llmtyped_sel_llmEv_${tag}_s${s}/DONE" ]; then
      copy_llm_mut "$ROOT/ga_llmtyped_pre_${tag}_s${s}" "$ROOT/ga_llmtyped_sel_llmEv_${tag}_s${s}" || { echo "!!! 复制失败，停止"; exit 1; }
    fi
    run "ga_llmtyped_sel_llmEv_${tag}" "$s" 200 --mutation llm_typed --select llm $EV --model "$model" \
        --init_from "$ROOT/ga_llmtyped_pre_${tag}_s${s}/evals.csv"
  done 3<<< "$MODELS"
done
while read -r tag model <&3; do
  [ -n "$MAX_USD" ] && uv run --locked python -m hyblock.ablation_cost --root "$ROOT" --tag "$tag" --model "$model" | tail -1
done 3<<< "$MODELS"
complete=1
for s in $ALL_SEEDS; do
  for tag in $TAGS; do [ -f "$ROOT/ga_llmtyped_sel_llmEv_${tag}_s${s}/DONE" ] || complete=0; done
done
if [ $complete -eq 1 ]; then
  uv run --locked python -m hyblock.model_ablation_analysis --base "$BASE" --root "$ROOT" --tags haiku $TAGS \
      --out "$MA/analysis" | tail -25
  echo "=== 全部完成 $(date '+%F %T')"
else
  echo "=== 这一批完成 $(date '+%F %T')：还有种子没跑完，按登记不做比较分析（10 个种子都跑完后再分析）"
fi
