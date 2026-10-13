"""Exercise the real dataset, all available models, forms and generated plots."""
import contextlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from evaluation import run_comparison
from tree_based import ALGORITHMS


def main():
    destination = ROOT / ".analysis-cache"
    destination.mkdir(exist_ok=True)
    report = {"checks": [], "dataset": "Policypulse.csv"}
    client = app.test_client()
    cases = [("GET", route, None) for route in ["/", "/data-loading", "/eda", "/preprocessing", "/linear-regression", "/logistic-regression"]]
    cases += [("POST", "/tree-based", {"algorithm": algorithm}) for algorithm in ALGORITHMS]
    cases += [("POST", "/unsupervised", {"method": method, "k": "3"}) for method in ["manual", "elbow", "silhouette"]]
    cases += [("POST", "/hierarchical", {"k": "3", "linkage": "ward"}), ("POST", "/dbscan", {"eps": "0.8", "min_samples": "10"}), ("POST", "/best-model", {})]
    with (destination / "verification.log").open("w", encoding="utf-8") as log:
        for method, path, fields in cases:
            started = time.perf_counter()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                response = client.open(path, method=method, data=fields)
            html = response.get_data(as_text=True)
            if response.status_code != 200 or '<div class="error">' in html:
                raise RuntimeError(f"{path} {fields}: {response.status_code}: {html[:1000]}")
            plots = re.findall(r'src="([^" ]+\.png(?:\?[^" ]*)?)"', html)
            for plot in plots:
                if client.get(plot).status_code != 200:
                    raise RuntimeError(f"Missing plot: {plot}")
            check = {"path": path, "form": fields, "seconds": round(time.perf_counter() - started, 2), "plots": len(plots), "status": "passed"}
            report["checks"].append(check)
            print(json.dumps(check), flush=True)
        results = run_comparison()
        report["classification"] = results["comparison"]
        report["regression"] = results["regression"]
        report["best_selected"] = results["best"]
        report["selection"] = results["selection"]
        report["training_rows"] = results["summary"]["training_rows"]
        report["testing_rows"] = results["summary"]["testing_rows"]
    (ROOT / "verification_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved verification_results.json", flush=True)


if __name__ == "__main__":
    main()
