"""A file written to a Task's folder in parts, of any content and any size (D-1207).

A caller sends a file's raw bytes a part at a time with `file_write_part`. Each part names the file's path, the part's
offset, and the whole file's size and SHA-256, so a write is named by its root Task, path, size and digest. The parts of
a write are appended, as they come, to one staged file in its root Task's folder of unfinished writes,
`.tapestry-parts/<root Task>/`. That folder lies beside the Task folders and outside every one of them, so no listing,
catalogue or read of a Task's files sees an unfinished write. It is on the same mount as the Task folders, so the
finished file takes its path by one rename.

Nothing holds a write's state between parts but its staged file: the bytes received are the staged file's size. A part
at offset 0 begins the write, and begins it again when an earlier attempt left bytes; each later part starts where the
bytes received end. When the bytes received reach the size, the staged file is hashed. With the stated digest it takes
the file's path, replacing any file there; with another it is removed. A write that never finishes leaves nothing under
its path. The server removes every unfinished write when it starts (`remove_unfinished_writes`); no timer or size bound
ends a write.
"""

from hashlib import file_digest, sha256
from os import O_APPEND, O_CLOEXEC, O_CREAT, O_NOFOLLOW, O_TRUNC, O_WRONLY, fdopen, fsync, open as os_open
from pathlib import Path
from re import fullmatch
from shutil import rmtree

# The folder beside the Task folders that holds, a folder per root Task, the staged file of each unfinished write.
PART_WRITE_FOLDERS = ".tapestry-parts"
# A staged file is readable and writable by the server's account alone, as a file `file_write` stages is.
STAGED_FILE_MODE = 0o600


class PartWriteRefusal(RuntimeError):
    """A part the write refuses, with the stable code and status its error answers and any further fields it names."""

    def __init__(self, message: str, code: str, status: int, details: dict):
        super().__init__(message)
        self.code = code
        self.status = status
        self.details = dict(details)


def require_part(offset: object, data: object, size: object, digest: object) -> None:
    """Refuse arguments that are not one part of one file: an offset and a size of 0 or more, raw bytes that end within
    the size and are empty only for an empty file, and the file's SHA-256 as 64 lowercase hex digits."""
    if (
        type(offset) is not int
        or type(size) is not int
        or type(data) is not bytes
        or type(digest) is not str
        or offset < 0
        or size < 0
        or not fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise PartWriteRefusal(
            "A part names an offset and a size of 0 or more, its raw bytes, and the file's SHA-256 as 64 lowercase hex digits",
            "bad_arguments",
            400,
            {},
        )
    if offset + len(data) > size:
        raise PartWriteRefusal("A part ends past the file's size", "bad_arguments", 400, {})
    if not data and size:
        raise PartWriteRefusal("A part of a file that is not empty carries bytes", "bad_arguments", 400, {})


def staged_name(relative: str, size: int, digest: str) -> str:
    """The staged file's name for the write of `size` bytes with SHA-256 `digest` to `relative` in its root Task's
    folder: the same for every part of one write, and another for any other write."""
    name = sha256(f"{relative}\0{size}\0{digest}".encode("utf-8", "surrogateescape")).hexdigest()
    return name


def write_part(folder: Path, target: Path, relative: str, offset: int, data: bytes, size: int, digest: str) -> dict:
    """Write the part `data` at `offset` of the file of `size` bytes with SHA-256 `digest` that goes to `relative` in
    the Task's folder, which resolves to `target`, staging it in `folder`, its root Task's folder of unfinished writes.
    Answer the bytes received, and once the last part is in, place the file whole at `target` (`place_file`). The
    arguments are a caller's, so each is checked (`require_part`) before it is used."""
    require_part(offset, data, size, digest)
    folder.mkdir(parents=True, exist_ok=True)
    if folder.resolve() != folder:
        raise PartWriteRefusal("The folder of unfinished writes is a link", "path_escape", 400, {})
    staged = folder / staged_name(relative, size, digest)
    try:
        received = staged.lstat().st_size
    except FileNotFoundError:
        received = 0
    if offset not in (0, received):
        raise PartWriteRefusal(
            f"The write holds {received} bytes: its next part starts there, or the write starts again at 0",
            "part_out_of_order",
            409,
            {"received": received},
        )
    flags = O_WRONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC | (O_TRUNC if offset == 0 else O_APPEND)
    descriptor = os_open(staged, flags, STAGED_FILE_MODE)
    with fdopen(descriptor, "wb") as staging:
        staging.write(data)
        staging.flush()
        fsync(staging.fileno())
    received = offset + len(data)
    if received < size:
        progress = {"path": relative, "received": received, "size": size, "complete": False}
        return progress
    place_file(staged, target, digest)
    written = {"path": relative, "received": received, "size": size, "sha256": digest, "complete": True}
    return written


def place_file(staged: Path, target: Path, digest: str) -> None:
    """Give the whole staged file `target`'s path when its SHA-256 is `digest`, replacing any file there by one rename;
    otherwise refuse the write. A staged file that is refused, or cannot take the path, is removed."""
    try:
        with staged.open("rb") as whole:
            found = file_digest(whole, "sha256").hexdigest()
        if found != digest:
            raise PartWriteRefusal(
                "The file's bytes have another SHA-256 than the write states; its parts were dropped",
                "digest_mismatch",
                422,
                {"sha256": found},
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(target)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def remove_unfinished_writes(workspace: Path, logger) -> None:
    """Remove every unfinished write under `workspace`, each one a stopped server's, so a write that never finished
    leaves nothing behind and none is removed by age or size.

    The server calls it before it accepts a request, so no write is in progress. Only a real folder is removed: a folder
    of unfinished writes that is a link is left as it is and named in the log.
    """
    folder = workspace / PART_WRITE_FOLDERS
    if folder.is_symlink() or folder.exists() and folder.resolve() != folder:
        logger.warning("unfinished writes folder %s is a link; it is left as it is", folder)
        return
    if not folder.is_dir():
        return
    removed = sum(1 for staged in folder.glob("*/*") if staged.is_file())
    rmtree(folder)
    logger.info("%s unfinished writes of a stopped server were removed", removed)
