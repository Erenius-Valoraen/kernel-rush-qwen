import ast
import gzip
import io
import tarfile
from pathlib import Path

ALLOWED_SUFFIXES = frozenset({
    ".py", ".pyi", ".yaml", ".yml", ".json", ".toml", ".txt", ".md", ".cfg", ".ini",
})
SKIP_DIRECTORIES = frozenset({
    "__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".ruff_cache", ".pytest_cache",
})
MAX_FILES = 200
MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024


def engine_files(engine_dir: Path) -> list[tuple[str, Path]]:
    """Every file that will ship, as ``(archive path, source path)``, sorted."""
    found = []
    for path in sorted(engine_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(engine_dir)
        if relative.as_posix() == "dryft.yaml":
            continue
        if any(part.startswith(".") or part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        found.append((relative.as_posix(), path))
    return found


def check(engine_dir: Path) -> list[tuple[str, Path]]:
    """Refuse what the platform would refuse, with the reason it would give."""
    files = engine_files(engine_dir)
    names = {name for name, _ in files}
    if "engine.py" not in names:
        raise ValueError(
            f"{engine_dir}/engine.py is missing: the archive root must hold it"
        )
    source = (engine_dir / "engine.py").read_text()
    classes = {
        node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
    }
    if "Engine" not in classes:
        raise ValueError(
            f"{engine_dir}/engine.py must export 'class Engine'; it defines "
            f"{sorted(classes) or 'no classes'}"
        )
    if len(files) > MAX_FILES:
        raise ValueError(f"{len(files)} files exceeds the limit of {MAX_FILES}")
    return files


def package(engine_dir: Path) -> bytes:
    """Check, then build a byte-for-byte reproducible ``.tar.gz``.

    Timestamps and ownership are zeroed and names are sorted, so an unchanged
    tree always produces the same archive and the same digest.
    """
    files = check(engine_dir)
    raw = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        total = 0
        for name, path in files:
            payload = path.read_bytes()
            total += len(payload)
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"the files expand to {total} bytes, over the "
                    f"{MAX_UNCOMPRESSED_BYTES} limit"
                )
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    built = raw.getvalue()
    if len(built) > MAX_COMPRESSED_BYTES:
        raise ValueError(
            f"the archive is {len(built)} bytes, over the {MAX_COMPRESSED_BYTES} limit"
        )
    return built
