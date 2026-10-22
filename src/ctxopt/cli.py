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


def _read_input(file: Optional[Path]) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    err_console.print("[red]No input file given and nothing piped on stdin.[/red]")
    raise typer.Exit(code=1)


@app.command()
def sandbox(
    file: Optional[Path] = typer.Argument(None, exists=True, help="File to read. Omit to read stdin."),
    max_lines: int = typer.Option(200, help="Collapse output once it exceeds this many lines."),
    head: int = typer.Option(40, help="Lines to keep from the start."),
    tail: int = typer.Option(40, help="Lines to keep from the end."),
    out: Optional[Path] = typer.Option(None, help="Write result to this file instead of stdout."),
) -> None:
    """Collapse noisy or oversized tool output into a compact form."""
    text = _read_input(file)
    result = sandbox_text(text, max_lines=max_lines, head_lines=head, tail_lines=tail)

    if out:
        out.write_text(result.text, encoding="utf-8")
    else:
        console.print(result.text)

    saved = result.original_tokens - result.final_tokens
    err_console.print(
        f"[dim]{result.original_tokens} -> {result.final_tokens} tokens "
        f"({saved} saved, {result.lines_elided} lines elided)[/dim]"
    )


@app.command()
def summarize(
    file: Optional[Path] = typer.Argument(None, exists=True, help="File to read. Omit to read stdin."),
    ratio: float = typer.Option(0.25, help="Target size as a fraction of the original."),
    out: Optional[Path] = typer.Option(None, help="Write result to this file instead of stdout."),
) -> None:
    """Condense a long conversation or log into a shorter digest."""
    text = _read_input(file)
    result = summarize_text(text, target_ratio=ratio)

    if out:
        out.write_text(result.text, encoding="utf-8")
    else:
        console.print(result.text)

    err_console.print(f"[dim]{result.original_tokens} -> {result.final_tokens} tokens[/dim]")


@app.command(name="route")
def route_command(
    prompt: str = typer.Argument(..., help="The prompt to score and route."),
) -> None:
    """Score a prompt's complexity and recommend a model tier."""
    decision = route_prompt(prompt, RouterConfig())

    table = Table(show_header=False, box=None)
    table.add_row("Score", f"{decision.score:.2f}")
    table.add_row("Tier", decision.tier)
    table.add_row("Model", decision.model)
    table.add_row("Reasons", "; ".join(decision.reasons) or "none")
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
