PYTHON ?= python3

.PHONY: test run-smoke lint reproduce-all seal math-audit lean-audit paper-assets paper paper-clean paper-flatten

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

run-smoke:
	PYTHONPATH=src $(PYTHON) -m emergent_logic.smoke

lint:
	PYTHONPATH=src ruff check .

reproduce-all:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_all.py

seal:
	$(MAKE) test
	$(MAKE) reproduce-all
	PYTHONPATH=src $(PYTHON) scripts/freeze_claims.py
	PYTHONPATH=src $(PYTHON) scripts/validate_final_state.py
	$(MAKE) math-audit
	$(MAKE) lean-audit

math-audit:
	$(PYTHON) scripts/audit_mathematics.py --output results/math_review/audit.json

lean-audit:
	$(PYTHON) scripts/check_lean_axioms.py --output results/math_review/lean_axioms.json

paper-assets:
	PYTHONPATH=src $(PYTHON) scripts/render_paper_assets.py

paper: paper-assets
	mkdir -p paper/build
	@if command -v latexmk >/dev/null 2>&1; then \
		cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex; \
	else \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; \
	fi
	$(MAKE) paper-flatten

paper-flatten:
	mkdir -p paper/build
	@if command -v latexpand >/dev/null 2>&1; then \
		cd paper && latexpand -o build/main_flat.tex main.tex; \
	else \
		echo "latexpand not found; cannot write paper/build/main_flat.tex" >&2; \
		exit 1; \
	fi

paper-clean:
	rm -rf paper/build
