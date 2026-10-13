"""Project-relative paths and serialized, dataset-aware analysis caching."""
from functools import wraps
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import hashlib
import os
import joblib
from importlib.metadata import version, PackageNotFoundError

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Policypulse.csv"
PLOT_DIR = BASE_DIR / "static" / "plots"
ANALYSIS_LOCK = RLock()
CACHE_SCHEMA = 2


def _library_versions():
    versions = []
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"]:
        try:
            versions.append((name, version(name)))
        except PackageNotFoundError:
            versions.append((name, None))
    return tuple(versions)


LIBRARY_VERSIONS = _library_versions()
SOURCE_SNAPSHOT = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in BASE_DIR.glob("*.py")}


@contextmanager
def atomic_output(destination):
    """Write a unique temporary file before replacing a completed artifact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".policypulse-", suffix=destination.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_manifest(result):
    manifest = {}
    if not isinstance(result, dict):
        return manifest
    for key, value in result.items():
        if key == "plots" or key.endswith("_plot"):
            values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str) and item.endswith(".png"):
                    relative = item[6:] if item.startswith("plots/") else item
                    path = PLOT_DIR / relative
                    manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return manifest


def _plots_match(result, manifest):
    current = _plot_manifest(result)
    return current == manifest and all(value is not None for value in current.values())


def dataset_version():
    stat = DATA_PATH.stat()
    return str(DATA_PATH), stat.st_mtime_ns, stat.st_size


def cached_analysis(function):
    """Avoid retraining on refresh and protect Matplotlib's shared state."""
    cache = {}
    manifests = {}

    @wraps(function)
    def wrapped(*args, **kwargs):
        key = (dataset_version(), args, tuple(sorted(kwargs.items())))
        with ANALYSIS_LOCK:
            if key in cache and not _plots_match(cache[key], manifests[key]):
                del cache[key]
            if key not in cache:
                if cache and next(iter(cache))[0] != key[0]:
                    cache.clear()
                # Model and plot changes invalidate disk results as well.
                dependencies = {"project_runtime.py", "preprocessing.py", "load_data.py", "modeling.py", function.__module__ + ".py"}
                if function.__module__ in {"evaluation", "model_selection"}:
                    dependencies.update({"linear_regression.py", "logistic_regression.py", "tree_based.py", "model_selection.py"})
                # Fingerprint the code loaded by this process. An older running
                # server must never sign its results with newer files on disk.
                source_version = tuple((name, SOURCE_SNAPSHOT.get(name)) for name in sorted(dependencies))
                digest = hashlib.sha256(repr((CACHE_SCHEMA, key, source_version, LIBRARY_VERSIONS)).encode()).hexdigest()[:20]
                destination = BASE_DIR / ".analysis-cache" / f"{function.__module__}-{function.__name__}-{digest}.joblib"
                if destination.exists():
                    try:
                        saved = joblib.load(destination)
                        if saved.get("schema") == CACHE_SCHEMA and _plots_match(saved["result"], saved["plots"]):
                            cache[key], manifests[key] = saved["result"], saved["plots"]
                    except (OSError, ValueError, EOFError, KeyError, AttributeError):
                        destination.unlink(missing_ok=True)
                if key not in cache:
                    cache[key] = function(*args, **kwargs)
                    manifests[key] = _plot_manifest(cache[key])
                    if not _plots_match(cache[key], manifests[key]):
                        del cache[key]
                        raise RuntimeError("Analysis produced results without all expected chart files.")
                    destination.parent.mkdir(exist_ok=True)
                    with atomic_output(destination) as temporary:
                        joblib.dump({"schema": CACHE_SCHEMA, "result": cache[key], "plots": manifests[key]}, temporary, compress=3)
            return cache[key]

    def clear():
        with ANALYSIS_LOCK:
            cache.clear()
            manifests.clear()
    wrapped.cache_clear = clear
    return wrapped
