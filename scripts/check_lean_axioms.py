"""Build Lean and audit every named theorem's transitive axioms."""

from pathlib import Path
import argparse
import json
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1] / "lean" / "LogicClosure"
    subprocess.run(["lake", "build"], cwd=project, check=True)
    names = []
    for source in sorted((project / "LogicClosure").glob("*.lean")):
        text = source.read_text()
        if re.search(r"\b(sorry|admit|axiom|unsafe)\b", text):
            raise RuntimeError(f"Unexpected proof escape in {source}")
        names += [
            f"LogicClosure.{name}"
            for name in re.findall(r"^theorem\s+(\w+)", text, flags=re.M)
        ]
    code = "import LogicClosure\n" + "\n".join(
        f"#print axioms {name}" for name in names
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", dir=project) as tmp:
        tmp.write(code)
        tmp.flush()
        p = subprocess.run(
            ["lake", "env", "lean", tmp.name],
            cwd=project,
            capture_output=True,
            text=True,
            check=True,
        )
    receipts = {}
    for name in names:
        match = re.search(
            r"'" + re.escape(name) + r"' depends on axioms: \[([^\]]*)\]", p.stdout
        )
        if match:
            axioms = [a.strip() for a in match.group(1).split(",") if a.strip()]
        elif f"'{name}' does not depend on any axioms" in p.stdout:
            axioms = []
        else:
            raise RuntimeError(f"Missing axiom receipt for {name}: {p.stdout}")
        if set(axioms) - {"propext", "Classical.choice", "Quot.sound"}:
            raise RuntimeError(f"Unexpected transitive axioms for {name}: {axioms}")
        receipts[name] = axioms
    report = {
        "named_theorems": len(receipts),
        "axioms": receipts,
        "toolchain": (project / "lean-toolchain").read_text().strip(),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
