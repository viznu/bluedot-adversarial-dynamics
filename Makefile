.PHONY: demo run-b analyze-latest

# 60-second live demo: 6 agents, 2 names, watch a convention form in the terminal
demo:
	uv run python exp_b_norm_capture/naming_game.py \
		--config exp_b_norm_capture/config.yaml \
		--seed 0 --population-size 6 --max-rounds 12 --run-id demo

# full experiment-B run (24 agents, to convergence)
run-b:
	uv run python exp_b_norm_capture/naming_game.py \
		--config exp_b_norm_capture/config.yaml --seed 7

# plot the most recent run
analyze-latest:
	uv run python exp_b_norm_capture/analyze.py \
		"$$(ls -t exp_b_norm_capture/runs/*.jsonl | head -1)"
