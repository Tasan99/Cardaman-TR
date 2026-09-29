"""Parse untrusted documents in a disposable, bounded subprocess."""
import multiprocessing
import sys

from .models import Document, Download, IngestionError


def _child(pipe, source: Download) -> None:
    try:
        if sys.platform == "linux":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
        from .parse import parse
        pipe.send((True, parse(source)))
    except Exception as exc:
        pipe.send((False, str(exc) if isinstance(exc, IngestionError) else "Parser failed; source requires review"))
    finally:
        pipe.close()


def bounded_parse(source: Download) -> Document:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(child, source))
    process.start()
    child.close()
    try:
        if not parent.poll(35):
            raise IngestionError("Parser exceeded time/memory budget; source requires review")
        try:
            success, result = parent.recv()
        except EOFError as exc:
            raise IngestionError("Parser exited without a result; source requires review") from exc
        if not success:
            raise IngestionError(result)
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
