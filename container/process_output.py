"""A finished tool process's output as the Workspace server answers it: read from disk a line at a time for a tool that
parses it, or kept on disk for its caller to read in parts when it is larger than the caller's read part (D-1145).

The server runs each tool process with its stdout and stderr written to anonymous files as they are produced. Nothing
here holds an output whole: a parser reads one line at a time, and a kept copy is made a block at a time.
"""

from base64 import b64decode
from codecs import getincrementaldecoder
from hashlib import sha256
from json import loads as json_loads
from os import fdopen, fsencode, fsync, scandir
from pathlib import Path
from posixpath import normpath, relpath
from shutil import rmtree
from tempfile import mkstemp

# The folder beside the Task folders that keeps, a folder per root Task, each process output larger than its caller's
# read part, named by the request and its stream. It lies outside every Task folder, so no listing, catalogue or change
# of a Task's files includes it. A root Task's kept outputs go when its Task folder goes (`remove_unowned_run_outputs`).
RUN_OUTPUT_FOLDERS = ".tapestry-output"
# The streams of a process whose output is kept, in the order a tool's outputs are held.
RUN_OUTPUT_STREAMS = ("stdout", "stderr")
# The name a kept output's copy has until it is whole and takes the output's own name.
KEPT_COPY_PREFIX = ".tapestry-output-"
# The field a reply names a kept output by, the shape a stage result names a large text by
# (tapestry/evidence/result_text.py).
TEXT_REFERENCE_FIELD = "text_by_reference"


def output_lines(output):
    """Each line of a finished output as text, read from disk one line at a time, invalid bytes replaced."""
    for line in output:
        yield line.decode("utf-8", errors="replace")


def text_lines(lines) -> list:
    """The non-empty lines of an output, each without its line ending, read one line at a time."""
    found = [line.rstrip("\r\n") for line in lines if line.strip()]
    return found


def json_document(lines):
    """The one JSON document an output holds."""
    document = json_loads("".join(lines))
    return document


def log_commits(lines) -> list:
    """The commit records of `git log --format=%H%x00%h%x00%aI%x00%s`, read one line at a time."""
    commits = []
    for line in lines:
        commit_hash, short_hash, authored_at, subject = line.rstrip("\n").split("\x00", 3)
        commits.append({"hash": commit_hash, "shortHash": short_hash, "authoredAt": authored_at, "subject": subject})
    return commits


def ripgrep_text(field: object) -> str:
    """Decode a ripgrep JSON text or base64 bytes field into UTF-8 text."""
    decoded = ""
    if isinstance(field, dict):
        text_value = field.get("text", "")
        bytes_value = field.get("bytes", "")
        if "text" in field and isinstance(text_value, str):
            decoded = text_value
        elif "bytes" in field and isinstance(bytes_value, str):
            decoded = b64decode(bytes_value).decode("utf-8", errors="replace")
    return decoded


def ripgrep_relative_path(field: object, base: Path) -> str:
    """Resolve a ripgrep JSON path field (`text` or base64 `bytes`) relative to `base`."""
    if isinstance(field, dict):
        text_value = field.get("text", "")
        if "text" in field and isinstance(text_value, str) and text_value:
            relative_text = str(Path(text_value).relative_to(base))
            return relative_text
        bytes_value = field.get("bytes", "")
        if "bytes" in field and isinstance(bytes_value, str):
            raw = b64decode(bytes_value)
            if not raw:
                empty_relative = str(Path("").relative_to(base))
                return empty_relative
            raw_path = normpath(raw)
            base_bytes = normpath(fsencode(base))
            if base_bytes == b".":
                in_base = not raw_path.startswith(b"/") and raw_path != b".." and not raw_path.startswith(b"../")
            else:
                prefix = b"/" if base_bytes == b"/" else base_bytes.rstrip(b"/") + b"/"
                in_base = raw_path == base_bytes or raw_path.startswith(prefix)
            if not in_base:
                raise ValueError(f"{raw!r} is not in the subpath of {str(base)!r}")
            relative_bytes = relpath(raw_path, base_bytes).decode("utf-8", errors="replace")
            return relative_bytes
    fallback = str(Path("").relative_to(base))
    return fallback


def search_matches(lines, max_results: int, base: Path) -> list:
    """The matches of ripgrep's JSON lines, read one line at a time, each file named relative to `base`: every match,
    or the first `max_results`."""
    matches = []
    for line in lines:
        data = json_loads(line) if line.strip() else {}
        if data.get("type", "") != "match":
            continue
        match = data.get("data", {})
        matches.append(
            {
                "file": ripgrep_relative_path(match.get("path", {}), base),
                "line": match.get("line_number", 0),
                "content": ripgrep_text(match.get("lines", {})).strip(),
            }
        )
        if max_results and len(matches) >= max_results:
            break
    return matches


def output_detail(value) -> str:
    """A failed run's output as its error message gives it: the text a read part holds, or where the larger kept output
    is read."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped
    reference = value.get(TEXT_REFERENCE_FIELD, {}) if isinstance(value, dict) else {}
    detail = f"its output of {reference.get('utf8_bytes', 0)} UTF-8 bytes is kept for run_output_read under this request"
    return detail


def keep_output(output, target: Path, copy_block_bytes: int) -> dict:
    """Keep a finished output at `target` and return its reference, `{"text_by_reference": {"sha256", "utf8_bytes"}}`.

    The output is kept as UTF-8, with invalid bytes replaced as a carried text replaces them. It is copied, hashed and
    counted a block at a time beside `target`. The copy takes `target`'s name, replacing a kept output of that name, only
    once it is whole and on disk.
    """
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    digest, size = sha256(), 0
    descriptor, temporary_name = mkstemp(prefix=KEPT_COPY_PREFIX, dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with fdopen(descriptor, "wb") as kept:
            while True:
                block = output.read(copy_block_bytes)
                encoded = decoder.decode(block, final=not block).encode("utf-8")
                digest.update(encoded)
                size += len(encoded)
                kept.write(encoded)
                if not block:
                    break
            kept.flush()
            fsync(kept.fileno())
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    reference = {TEXT_REFERENCE_FIELD: {"sha256": digest.hexdigest(), "utf8_bytes": size}}
    return reference


def remove_unowned_run_outputs(workspace: Path, task_folders: str, logger) -> None:
    """Remove the kept outputs of each root Task whose Task folder under `workspace` is gone, and any copy a stopped
    server left before it was whole, so a root Task's kept outputs go with its Task folder and never by age or count.

    The server calls it before it accepts a request, so no root Task is reading one. Only real folders are removed: an
    output folder that is a link is left as it is and named in the log.
    """
    outputs = workspace / RUN_OUTPUT_FOLDERS
    if outputs.is_symlink() or outputs.exists() and outputs.resolve() != outputs:
        logger.warning("kept outputs folder %s is a link; its kept outputs are left as they are", outputs)
        return
    if not outputs.is_dir():
        return
    removed = 0
    with scandir(outputs) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if (workspace / task_folders / entry.name).is_dir():
                with scandir(entry.path) as kept:
                    for copy in kept:
                        if copy.name.startswith(KEPT_COPY_PREFIX) and copy.is_file(follow_symlinks=False):
                            Path(copy.path).unlink(missing_ok=True)
            else:
                rmtree(entry.path)
                removed += 1
    logger.info("kept outputs of %s root Tasks whose Task folder is gone were removed", removed)
