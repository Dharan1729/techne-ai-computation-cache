#!/usr/bin/env bash
# Reproduce the headline results. Run from the repository root.
#
#   bash scripts/reproduce.sh quick     # tests + wrong-hit guard comparison   (~5 min, offline)
#   bash scripts/reproduce.sh sweep     # repetition sweep + figure 2           (~15 min, offline)
#   bash scripts/reproduce.sh realmodel # fair-baseline comparison, figure 1    (~15 min, needs Ollama + llama3)
#   bash scripts/reproduce.sh realapp   # financial-chat integration            (~10 min, needs Ollama + llama3, network)
#
# Everything writes to out/ and never overwrites the committed results/.
# The first run downloads the embedding model (redis/langcache-embed-v1) from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out

case "${1:-quick}" in
  quick)
    python -m pytest -q
    python -m benchmarks.guard_ab
    ;;
  sweep)
    # Repetition sweep: how often a request repeats earlier numbers vs. what each cache can serve.
    for cfg in "18 1" "9 2" "6 3" "3 6"; do
      set -- $cfg
      for s in 6 7 8 9 10; do
        python -m benchmarks.run_experiment --backend simulated --seed "$s" \
          --param-sets-per-task "$1" --paraphrases-per-param-set "$2" \
          --skip-threshold-sweep --skip-external --output-dir "out/rep/P$1K$2s$s" > /dev/null
      done
    done
    python -m benchmarks.plot_linkedin --rep-dir out/rep --out out/figures
    echo "wrote out/figures/fig2_repeats.png"
    ;;
  realmodel)
    # Needs `ollama serve` running and `ollama pull llama3`.
    python -m benchmarks.real_baselines --model llama3 --seed 42 --output-dir out/real_baselines
    ;;
  realapp)
    # Needs `ollama serve` + llama3, and network access to clone financial-chat.
    [ -d out/financial-chat ] || git clone --depth 1 https://github.com/wshobson/financial-chat.git out/financial-chat
    python -m benchmarks.real_app.financial_chat_integration \
      --repo-path out/financial-chat --model llama3 --seed 7 --output out/real_app_financial_chat.json
    ;;
  *)
    echo "usage: bash scripts/reproduce.sh {quick|sweep|realmodel|realapp}" >&2
    exit 2
    ;;
esac
