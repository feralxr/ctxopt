"""Command-line interface for ctxopt."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .router import RouterConfig
from .router import route as route_prompt
from .safety import UnsafePatternError
from .sandbox import sandbox_text
from .summarize import summarize_text

app = typer.Typer(
    name="ctxopt",
    help=(
        "Context window optimizer: sandbox noisy tool output, summarize long "
        "conversations, and route prompts to the right model tier."
    ),
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

_DEFAULT_MAX_READ_BYTES = 20_000_000  # 20 MB


def _read_input(file: Optional[Path], max_input_bytes: int = _DEFAULT_MAX_READ_BYTES) -> str:
    if file is not None:
        size = file.stat().st_size
        if size > max_input_bytes:
            return _read_bounded_file(file, max_input_bytes)
        return file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return _read_capped_stdin(max_input_bytes)
    err_console.print("[red]No input file given and nothing piped on stdin.[/red]")
    raise typer.Exit(code=1)


def _read_bounded_file(path: Path, max_bytes: int) -> str:
    """Read only a bounded head+tail slice of a very large file via
    seeking, instead of loading the whole file into memory. sandbox and
    summarize only ever use a bounded head/tail window plus a scan of the
    middle for keep-patterns anyway, so for a file far larger than any
    reasonable budget could use, this avoids paying full-file memory and
    I/O cost for content that would just get elided right back out."""
    head_bytes = int(max_bytes * 0.6)
    tail_bytes = max_bytes - head_bytes
    total_size = path.stat().st_size

    with open(path, "rb") as f:
        head = f.read(head_bytes)
        if total_size > head_bytes + tail_bytes:
            f.seek(max(0, total_size - tail_bytes))
            tail = f.read()
        else:
            tail = b""

    marker = (
        f"\n... [file too large to read in full: kept first {len(head)} and "
        f"last {len(tail)} bytes of {total_size} total; raise --max-input-bytes "
        f"to read more] ...\n"
    ).encode()
    return (head + marker + tail).decode("utf-8", errors="replace")


def _read_capped_stdin(max_bytes: int) -> str:
    """Read at most ``max_bytes`` from stdin rather than buffering an
    unbounded pipe fully into memory. Unlike a file, a pipe can't be
    seeked, so this can't keep a head *and* tail window the way
    :func:`_read_bounded_file` does — it simply stops at the cap, then
    drains and discards whatever's left so an upstream writer doesn't block
    on a full pipe, rather than holding the remainder in memory too."""
    buf = sys.stdin.buffer
    data = buf.read(max_bytes + 1)  # +1 so we can tell "there was more" apart from "exactly at the cap"
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace")

    truncated = data[:max_bytes]
    discarded = len(data) - max_bytes  # the lookahead byte(s) already read beyond the cap
    while True:
        chunk = buf.read(1_000_000)
        if not chunk:
            break
        discarded += len(chunk)

    marker = (
        f"\n... [stdin truncated at {max_bytes} bytes; {discarded} more bytes "
        f"discarded; raise --max-input-bytes to read more] ...\n"
    )
    return truncated.decode("utf-8", errors="replace") + marker


@app.command()
def sandbox(
    file: Optional[Path] = typer.Argument(None, exists=True, help="File to read. Omit to read stdin."),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Shrink output to fit this token budget (recommended over --max-lines)."
    ),
    max_lines: int = typer.Option(
        200, help="Collapse output once it exceeds this many lines (ignored if --max-tokens is set)."
    ),
    head: Optional[int] = typer.Option(
        None, help="Lines to keep from the start. Defaults to a profile-specific value."
    ),
    tail: Optional[int] = typer.Option(
        None, help="Lines to keep from the end. Defaults to a profile-specific value."
    ),
    profile: str = typer.Option("auto", help="Content profile: auto, generic, json, stacktrace, or diff."),
    keep: list[str] = typer.Option([], "--keep", help="Extra regex pattern to always preserve (repeatable)."),
    out: Optional[Path] = typer.Option(None, help="Write result to this file instead of stdout."),
    max_input_bytes: int = typer.Option(
        _DEFAULT_MAX_READ_BYTES,
        "--max-input-bytes",
        help="Read at most this many bytes from a huge file/stdin, head+tail, rather than loading it all.",
    ),
) -> None:
    """Collapse noisy or oversized tool output into a compact form."""
    text = _read_input(file, max_input_bytes)
    try:
        result = sandbox_text(
            text,
            max_tokens=max_tokens,
            max_lines=max_lines,
            head_lines=head,
            tail_lines=tail,
            keep_patterns=list(keep) or None,
            profile=profile,
        )
    except UnsafePatternError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if out:
        out.write_text(result.text, encoding="utf-8")
    else:
        console.print(result.text)

    saved = result.original_tokens - result.final_tokens
    budget_note = ", fit to budget" if result.fit_to_budget else ""
    err_console.print(
        f"[dim]profile={result.profile}  {result.original_tokens} -> {result.final_tokens} tokens "
        f"({saved} saved, {result.lines_elided} lines elided{budget_note})[/dim]"
    )


@app.command()
def summarize(
    file: Optional[Path] = typer.Argument(None, exists=True, help="File to read. Omit to read stdin."),
    ratio: float = typer.Option(
        0.25, help="Target size as a fraction of the original (ignored if --max-tokens is set)."
    ),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Target size in tokens directly."),
    keep: list[str] = typer.Option([], "--keep", help="Extra regex pattern to always preserve (repeatable)."),
    keep_last: int = typer.Option(
        2, "--keep-last", help="Always keep this many of the most recent turns (conversations only)."
    ),
    llm: bool = typer.Option(
        False, "--llm", help="Use an LLM for abstractive summarization instead of the local heuristic."
    ),
    model: str = typer.Option("claude-haiku-5-5", help="Model to use with --llm."),
    out: Optional[Path] = typer.Option(None, help="Write result to this file instead of stdout."),
    max_input_bytes: int = typer.Option(
        _DEFAULT_MAX_READ_BYTES,
        "--max-input-bytes",
        help="Read at most this many bytes from a huge file/stdin, head+tail, rather than loading it all.",
    ),
) -> None:
    """Condense a long conversation or log into a shorter digest."""
    text = _read_input(file, max_input_bytes)

    llm_call = None
    if llm:
        from .llm import anthropic_backend

        llm_call = anthropic_backend(model=model)

    try:
        result = summarize_text(
            text,
            target_ratio=ratio,
            max_tokens=max_tokens,
            keep_patterns=list(keep) or None,
            keep_last_n_turns=keep_last,
            llm_call=llm_call,
        )
    except (RuntimeError, UnsafePatternError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if out:
        out.write_text(result.text, encoding="utf-8")
    else:
        console.print(result.text)

    turns_note = f"  turns {result.turns_kept}/{result.turns_total}" if result.turns_total is not None else ""
    err_console.print(
        f"[dim]backend={result.backend}  {result.original_tokens} -> {result.final_tokens} tokens{turns_note}[/dim]"
    )


@app.command(name="route")
def route_command(
    prompt: str = typer.Argument(..., help="The prompt to score and route."),
    history_tokens: int = typer.Option(0, "--history-tokens", help="Accumulated conversation size so far, if any."),
    output_tokens: int = typer.Option(500, "--output-tokens", help="Estimated response length, for cost estimation."),
    max_cost: Optional[float] = typer.Option(
        None, "--max-cost", help="Budget in USD; downgrades tier if the chosen tier exceeds it."
    ),
) -> None:
    """Score a prompt's complexity and recommend a model tier."""
    config = RouterConfig(max_cost_per_call=max_cost)
    decision = route_prompt(prompt, config, history_tokens=history_tokens, estimated_output_tokens=output_tokens)

    table = Table(show_header=False, box=None)
    table.add_row("Score", f"{decision.score:.2f}")
    table.add_row("Tier", decision.tier)
    table.add_row("Model", decision.model)
    if decision.estimated_cost_usd is not None:
        table.add_row("Est. cost", f"${decision.estimated_cost_usd:.5f}")
    if decision.estimated_latency_ms is not None:
        table.add_row("Est. latency", f"{decision.estimated_latency_ms:.0f} ms")
    table.add_row("Reasons", "; ".join(decision.reasons) or "none")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
