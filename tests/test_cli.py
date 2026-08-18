"""Tests for the CLI (part of task 4.11).

The contract under test is DEC-019's three exit codes — 0 affirmative, 1 ran-but-negative, 2
could-not-run — asserted on both `index` and `status` across all three states, plus `--json`
emitting the same structured dict the underlying function already returns.

``main()`` returns its exit code rather than calling ``sys.exit``, so these call it directly and
read stdout/stderr through ``capsys``. Indexing runs on the fake embedder.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from support import FakeEmbeddingProvider, configured_backend

from langgraph_context_mcp import cli
from langgraph_context_mcp.cli import EXIT_ERROR, EXIT_NEGATIVE, EXIT_OK, main
from langgraph_context_mcp.tools import mcp_tools

FIXTURE = Path(__file__).parent / "fixtures" / "sample_graphs" / "simple_graph.py"


@pytest.fixture(autouse=True)
def fake_embedder_everywhere():
    mcp_tools.set_embedder(FakeEmbeddingProvider())
    yield
    mcp_tools.set_embedder(None)


@pytest.fixture
def empty_repo(tmp_path):
    """A readable directory with Python in it but no LangGraph usage."""
    repo = tmp_path / "no_langgraph"
    repo.mkdir()
    (repo / "plain.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    return repo


# --------------------------------------------------------------------------------------------
# index — the three exit codes
# --------------------------------------------------------------------------------------------
def test_index_exits_0_when_graphs_are_found(fixture_repo, capsys):
    code = main(["index", str(fixture_repo)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "1 graph(s), 5 node(s), 8 edge(s)" in out


def test_index_exits_1_when_no_graphs_are_found(empty_repo, capsys):
    """DEC-019: a negative answer, not a failure — the message says so."""
    code = main(["index", str(empty_repo)])

    assert code == EXIT_NEGATIVE
    out = capsys.readouterr().out
    assert "No LangGraph graphs found" in out


def test_index_exits_2_when_the_path_does_not_exist(tmp_path, capsys):
    code = main(["index", str(tmp_path / "missing")])

    assert code == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_index_exits_2_when_the_path_is_a_file(tmp_path, capsys):
    file_path = tmp_path / "a_file.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    assert main(["index", str(file_path)]) == EXIT_ERROR
    assert "not a directory" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("bad_path", ["", "   "])
def test_index_exits_2_on_an_empty_path(bad_path, capsys):
    """QA-3-07 at the CLI boundary: an empty path must not index the working directory."""
    assert main(["index", bad_path]) == EXIT_ERROR
    assert "Invalid path" in capsys.readouterr().err


def test_index_exits_2_on_parent_traversal(fixture_repo, capsys):
    """QA-3-08 / DEC-014 at the CLI boundary."""
    assert main(["index", str(fixture_repo / ".." / "..")]) == EXIT_ERROR
    assert "'..'" in capsys.readouterr().err


def test_index_reports_partial_nodes_when_present(tmp_path, capsys):
    repo = tmp_path / "dynamic"
    repo.mkdir()
    (repo / "agent.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "\n"
        "def make_node(tag):\n"
        "    def inner(state): ...\n"
        "    return inner\n"
        "\n"
        "graph = StateGraph(dict)\n"
        "graph.add_node('built', make_node('x'))\n",
        encoding="utf-8",
    )

    code = main(["index", str(repo)])

    assert code == EXIT_OK
    assert "partially resolved" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# status — the three exit codes
# --------------------------------------------------------------------------------------------
def test_status_exits_1_before_indexing(fixture_repo, capsys):
    code = main(["status", str(fixture_repo)])

    assert code == EXIT_NEGATIVE
    out = capsys.readouterr().out
    assert "No index found" in out
    assert "index" in out  # tells the user what to run next


def test_status_exits_0_after_indexing(fixture_repo, capsys):
    main(["index", str(fixture_repo)])
    capsys.readouterr()

    code = main(["status", str(fixture_repo)])

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "Index found" in out
    assert "1 graph(s), 5 node(s)" in out
    assert configured_backend() in out


def test_status_exits_2_when_the_path_does_not_exist(tmp_path, capsys):
    assert main(["status", str(tmp_path / "missing")]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_status_exits_2_on_parent_traversal(fixture_repo, capsys):
    assert main(["status", str(fixture_repo / "..")]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_status_does_not_create_an_index(fixture_repo):
    """Asking a question must not write into the repository (DEC-018)."""
    main(["status", str(fixture_repo)])

    assert not (fixture_repo / ".langgraph-context").exists()


# --------------------------------------------------------------------------------------------
# --json
# --------------------------------------------------------------------------------------------
def test_index_json_matches_the_index_repo_contract(fixture_repo, capsys):
    code = main(["index", str(fixture_repo), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert set(payload) == {
        "graphs_found",
        "nodes_indexed",
        "edges_indexed",
        "partial_nodes",
        "backend",
        "duration_ms",
    }
    assert payload["graphs_found"] == 1


def test_index_json_keeps_the_negative_exit_code(empty_repo, capsys):
    """--json changes the rendering, never the exit code."""
    code = main(["index", str(empty_repo), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_NEGATIVE
    assert payload["graphs_found"] == 0


def test_status_json_reports_the_structured_status(fixture_repo, capsys):
    main(["index", str(fixture_repo)])
    capsys.readouterr()

    code = main(["status", str(fixture_repo), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["indexed"] is True
    assert payload["graph_count"] == 1
    assert payload["node_count"] == 5
    assert payload["backend"] == configured_backend()


def test_status_json_when_not_indexed(fixture_repo, capsys):
    code = main(["status", str(fixture_repo), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_NEGATIVE
    assert payload["indexed"] is False


def test_json_output_is_the_only_thing_on_stdout(fixture_repo, capsys):
    """A caller piping --json into a parser must not receive prose alongside it."""
    main(["index", str(fixture_repo), "--json"])

    json.loads(capsys.readouterr().out)  # raises if anything else was printed


# --------------------------------------------------------------------------------------------
# Argument parsing and serve
# --------------------------------------------------------------------------------------------
def test_serve_takes_no_json_flag():
    """--json on serve would put bytes on the MCP transport — argparse must reject it."""
    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--json"])

    assert excinfo.value.code == EXIT_ERROR


def test_a_missing_subcommand_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == EXIT_ERROR


def test_an_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate", "."])

    assert excinfo.value.code == EXIT_ERROR


def test_index_requires_a_path():
    with pytest.raises(SystemExit) as excinfo:
        main(["index"])

    assert excinfo.value.code == EXIT_ERROR


def test_serve_returns_0_on_keyboard_interrupt(monkeypatch):
    """Ctrl-C is a clean shutdown, not a failure (scenario 5.16's contract)."""
    monkeypatch.setattr(
        "langgraph_context_mcp.cli.run_stdio",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert main(["serve"]) == EXIT_OK


def test_serve_returns_2_when_it_cannot_start(monkeypatch, capsys):
    monkeypatch.setattr(
        "langgraph_context_mcp.cli.run_stdio",
        lambda: (_ for _ in ()).throw(RuntimeError("backend unreachable")),
    )

    assert main(["serve"]) == EXIT_ERROR
    assert "backend unreachable" in capsys.readouterr().err


def test_serve_writes_nothing_to_stdout(monkeypatch, capsys):
    """The MCP stdio transport owns stdout — a single stray byte corrupts the stream."""
    monkeypatch.setattr("langgraph_context_mcp.cli.run_stdio", lambda: None)

    main(["serve"])

    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------------------------
# QA-5-05 — non-ASCII repo paths must not crash the human-readable output path
# --------------------------------------------------------------------------------------------
# The two scripts QA bisected the failure with. Latin-1 names never reproduced it, so they are not
# a regression test; these two are.
NON_ASCII_DIR_NAMES = ["日本語", "проект"]


class _LegacyConsole(io.TextIOBase):
    """A stream that fails exactly the way a Windows cp1252 console does.

    The real bug is invisible on a UTF-8 terminal, so reproducing it needs a stream whose encoding
    genuinely cannot represent the path — otherwise the test passes everywhere and guards nothing.
    """

    encoding = "cp1252"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, s: str) -> int:
        s.encode(self.encoding)  # raises UnicodeEncodeError on CJK / Cyrillic
        self.written.append(s)
        return len(s)

    @property
    def text(self) -> str:
        return "".join(self.written)


@pytest.mark.parametrize("dir_name", NON_ASCII_DIR_NAMES)
def test_index_and_status_survive_a_non_ascii_path_on_a_legacy_console(
    tmp_path, monkeypatch, dir_name
):
    """QA-5-05: indexing succeeded, then printing the result crashed with UnicodeEncodeError.

    Both subcommands are exercised because both interpolate the repo path into their summary.
    """
    repo = tmp_path / dir_name
    repo.mkdir()
    (repo / "graph.py").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    console = _LegacyConsole()
    monkeypatch.setattr(sys, "stdout", console)
    index_code = main(["index", str(repo)])
    status_code = main(["status", str(repo)])

    assert index_code == EXIT_OK, "the index itself always worked; only printing it failed"
    assert status_code == EXIT_OK
    # Not lost to '?' — the directory must stay identifiable in the escaped form.
    assert "\\u" in console.text


@pytest.mark.parametrize("dir_name", NON_ASCII_DIR_NAMES)
def test_non_ascii_path_output_is_exact_when_the_console_can_encode_it(
    tmp_path, capsys, dir_name
):
    """The fallback must be a fallback: a UTF-8 stream still gets the real characters."""
    repo = tmp_path / dir_name
    repo.mkdir()
    (repo / "graph.py").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    assert main(["index", str(repo)]) == EXIT_OK
    assert dir_name in capsys.readouterr().out


def test_a_failure_in_the_output_stage_exits_2_not_1(fixture_repo, monkeypatch, capsys):
    """The second half of QA-5-05, and the reason it was a contract bug and not only a crash.

    An exception escaping `main()` ends the process with Python's default exit code of 1 — the
    same number DEC-019 reserves for "ran correctly, the answer is negative". A script would read
    a crash as a clean negative answer. `main()` must therefore *return* 2 rather than raise.
    """
    def explode(*args, **kwargs):
        raise UnicodeEncodeError("cp1252", "x", 0, 1, "simulated console failure")

    monkeypatch.setattr(cli, "_write_line", explode)

    code = main(["index", str(fixture_repo)])

    assert code == EXIT_ERROR, "a failure must land on 2 by contract, never on 1 by accident"
    assert code != EXIT_NEGATIVE


def test_main_returns_rather_than_raising_when_output_fails(fixture_repo, monkeypatch):
    """Pin the mechanism, not just the number: nothing may escape `main()`.

    If an exception ever escapes again, the process exit code silently becomes 1 regardless of
    what the contract says — so this asserts the absence of a raise directly.
    """
    def explode(*args, **kwargs):
        raise RuntimeError("unexpected failure inside the command")

    monkeypatch.setattr(cli, "_run_index", explode)

    try:
        code = main(["index", str(fixture_repo)])
    except Exception as exc:  # noqa: BLE001 - the failure mode under test
        pytest.fail(f"main() raised {type(exc).__name__} instead of returning an exit code")

    assert code == EXIT_ERROR
