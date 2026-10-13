"""Project-relative paths and serialized, dataset-aware analysis caching."""
from functools import wraps
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import hashlib
import os
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Policypulse.csv"
PLOT_DIR = BASE_DIR / "static" / "plots"
ANALYSIS_LOCK = RLock()


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


def _plots_exist(result):
    if not isinstance(result, dict):
        return True
    for key, value in result.items():
        if key == "plots" or key.endswith("_plot"):
            values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
            if any(isinstance(item, str) and item.endswith(".png") and not (PLOT_DIR / Path(item).name).exists() for item in values):
                return False
    return True


def dataset_version():
    stat = DATA_PATH.stat()
    return str(DATA_PATH), stat.st_mtime_ns, stat.st_size


def cached_analysis(function):
    """Avoid retraining on refresh and protect Matplotlib's shared state."""
    cache = {}

    @wraps(function)
    def wrapped(*args, **kwargs):
        key = (dataset_version(), args, tuple(sorted(kwargs.items())))
        with ANALYSIS_LOCK:
            if key not in cache:
                if cache and next(iter(cache))[0] != key[0]:
                    cache.clear()
                # Model and plot changes invalidate disk results as well.
                dependencies = {"project_runtime.py", "preprocessing.py", "load_data.py", "modeling.py", function.__module__ + ".py"}
                if function.__module__ == "evaluation":
                    dependencies.update({"linear_regression.py", "logistic_regression.py", "tree_based.py"})
                source_version = tuple((name, hashlib.sha256((BASE_DIR / name).read_bytes()).hexdigest()) for name in sorted(dependencies) if (BASE_DIR / name).exists())
                digest = hashlib.sha256(repr((key, source_version)).encode()).hexdigest()[:20]
                destination = BASE_DIR / ".analysis-cache" / f"{function.__module__}-{function.__name__}-{digest}.joblib"
                if destination.exists():
                    try:
                        cache[key] = joblib.load(destination)
                        if not _plots_exist(cache[key]):
                            del cache[key]
                    except (OSError, ValueError, EOFError):
                        destination.unlink(missing_ok=True)
                if key not in cache:
                    cache[key] = function(*args, **kwargs)
                    destination.parent.mkdir(exist_ok=True)
                    with atomic_output(destination) as temporary:
                        joblib.dump(cache[key], temporary, compress=3)
            return cache[key]

    wrapped.cache_clear = cache.clear
    return wrapped
