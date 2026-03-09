.PHONY: test run-smoke lint reproduce-all seal paper-assets paper paper-clean

test:
	PYTHONPATH=src python -m pytest -q

run-smoke:
	PYTHONPATH=src python -m emergent_logic.smoke

lint:
	PYTHONPATH=src ruff check .

reproduce-all:
	PYTHONPATH=src python scripts/reproduce_all.py

seal:
	$(MAKE) test
	$(MAKE) reproduce-all
	PYTHONPATH=src python scripts/freeze_claims.py
	PYTHONPATH=src python scripts/validate_final_state.py
	cd lean/LogicClosure && lake build

paper-assets:
	PYTHONPATH=src python scripts/render_paper_assets.py

paper: paper-assets
	mkdir -p paper/build
	@if command -v latexmk >/dev/null 2>&1; then \
		cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex; \
	else \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; \
	fi

paper-clean:
	rm -rf paper/build
