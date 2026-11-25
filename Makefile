.PHONY: eval paper paper-body clean

# Run the full benchmark: rewrites results/metrics.json,
# results/per_query_ranks.json and results/summary_tables.md.
eval:
	python src/evaluate.py

# Compile the LaTeX body of the paper into paper/build/ (8 pages, no cover).
# The committed paper/research_paper.pdf is body + designed cover (merged);
# compiling into build/ keeps that final PDF from being overwritten.
paper-body:
	mkdir -p paper/build
	cd paper && tectonic --outdir build research_paper.tex

clean:
	rm -rf src/__pycache__ paper/*.aux paper/build
