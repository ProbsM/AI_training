from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# ----------------------------
# Config
# ----------------------------
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache"}
INCLUDE_EXTS = {".py"}

# Where your index lives (adjust if needed)
input("you'll need to update all the 'path_to_src' and 'src_ver' variables with the corresponding dir paths")
INDEX_PATH = Path(
    r"path_to_src"
    r"\path_to_src\src_ver\code_index.json"
)

# Root folder used for "grep" style tools (find_in_files, call sites, trace)
# We assume it's the directory containing code_index.json.
CODE_ROOT = INDEX_PATH.parent

# In-memory cache for the index
_INDEX_CACHE: list[dict] | None = None


# ----------------------------
# Index load/reload
# ----------------------------
def load_index() -> list[dict]:
    """Load the JSON symbol index into memory (with caching)."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            _INDEX_CACHE = json.load(f)
    return _INDEX_CACHE


@tool
def reload_index() -> str:
    """Reload code_index.json from disk (use after regenerating the index)."""
    global _INDEX_CACHE
    _INDEX_CACHE = None
    load_index()
    return f"Reloaded index from {INDEX_PATH}"


# ----------------------------
# Shared helpers
# ----------------------------
def iter_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in INCLUDE_EXTS:
                files.append(p)
    return files


def _read_text(p: Path) -> str:
    # utf-8-sig handles BOM (your earlier U+FEFF warnings)
    return p.read_text(encoding="utf-8-sig", errors="replace")


def _src_line(lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # py3.9+
    except Exception:
        return "..."


def _normalize_target(target: str) -> str:
    return (target or "").strip()


def _match_target_name(node: ast.AST, target: str) -> bool:
    """
    Match assignment targets against a target string.
    Supports:
      - "foo"
      - "self.foo"
      - "obj.foo"
    """
    if not target:
        return False

    # target like "self.data_save_path"
    if "." in target:
        parts = target.split(".")
        if len(parts) == 2:
            obj, attr = parts
            if isinstance(node, ast.Attribute) and node.attr == attr:
                # node.value must match obj (usually Name: self)
                if isinstance(node.value, ast.Name) and node.value.id == obj:
                    return True
            return False

    # target like "data_save_path"
    if isinstance(node, ast.Name) and node.id == target:
        return True

    # allow matching just attribute name "data_save_path" even if "self.data_save_path"
    if isinstance(node, ast.Attribute) and node.attr == target:
        return True

    return False


def _called_name(call: ast.Call) -> str:
    """
    Best-effort function name for a Call node:
      foo(...)            -> "foo"
      obj.foo(...)        -> "foo"
      pkg.mod.foo(...)    -> "foo"
    """
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


# ----------------------------
# Tools: search/open/grep/trace
# ----------------------------
@tool
def search_symbols_multi(queries: list[str], max_results_per_query: int = 6) -> list[dict]:
    """
    Run multiple searches over the code index and return merged, de-duplicated results.
    Each result contains: kind, name, qualname, file, lineno, signature, docstring.
    """
    symbols = load_index()

    seen = set()
    merged: list[dict] = []

    for q in queries or []:
        q = (q or "").lower().strip()
        if not q:
            continue

        local: list[dict] = []
        for sym in symbols:
            text = (
                f"{sym.get('name','')} "
                f"{sym.get('qualname','')} "
                f"{sym.get('docstring','')}"
            ).lower()
            if q in text:
                local.append(sym)

        for sym in local[:max_results_per_query]:
            key = f"{sym.get('file')}:{sym.get('lineno')}:{sym.get('qualname')}"
            if key not in seen:
                seen.add(key)
                merged.append(sym)

    return merged


@tool
def open_code(file_path: str, start_line: int, end_line: int) -> str:
    """
    Return lines from a source file (1-indexed, inclusive).
    Use this to inspect implementations after search.
    """
    p = Path(file_path)
    lines = _read_text(p).splitlines()

    start = max(1, int(start_line))
    end = min(len(lines), int(end_line))

    snippet = []
    for i in range(start, end + 1):
        snippet.append(f"{i:5d}: {lines[i-1]}")
    return "\n".join(snippet)


@tool
def find_in_files(pattern: str, case_sensitive: bool = False, max_results: int = 25) -> list[dict]:
    """
    Grep-like search through .py files under CODE_ROOT.
    Returns list of {file, lineno, line}.
    """
    pat = pattern or ""
    if not pat.strip():
        return []

    flags = 0 if case_sensitive else re.IGNORECASE

    results: list[dict] = []
    for fp in iter_py_files(CODE_ROOT):
        try:
            text = _read_text(fp)
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            if re.search(pat, line, flags=flags):
                results.append({"file": str(fp), "lineno": i, "line": line.strip()})
                if len(results) >= int(max_results):
                    return results

    return results


@tool
def trace_assignments(target: str, max_results: int = 25) -> list[dict]:
    """
    Find where a variable/attribute is assigned in the codebase via AST.
    target examples:
      - "data_save_path"
      - "self.data_save_path"
    Returns list of {file, lineno, kind, snippet}
      kind in {"assign", "annassign", "augassign"}
    """
    tgt = _normalize_target(target)
    if not tgt:
        return []

    results: list[dict] = []

    for fp in iter_py_files(CODE_ROOT):
        try:
            src = _read_text(fp)
            lines = src.splitlines()
            tree = ast.parse(src)
        except Exception:
            continue

        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if _match_target_name(t, tgt):
                        results.append({
                            "file": str(fp),
                            "lineno": getattr(n, "lineno", 0),
                            "kind": "assign",
                            "snippet": _src_line(lines, getattr(n, "lineno", 0)),
                        })
                        if len(results) >= int(max_results):
                            return results

            elif isinstance(n, ast.AnnAssign):
                if _match_target_name(n.target, tgt):
                    results.append({
                        "file": str(fp),
                        "lineno": getattr(n, "lineno", 0),
                        "kind": "annassign",
                        "snippet": _src_line(lines, getattr(n, "lineno", 0)),
                    })
                    if len(results) >= int(max_results):
                        return results

            elif isinstance(n, ast.AugAssign):
                if _match_target_name(n.target, tgt):
                    results.append({
                        "file": str(fp),
                        "lineno": getattr(n, "lineno", 0),
                        "kind": "augassign",
                        "snippet": _src_line(lines, getattr(n, "lineno", 0)),
                    })
                    if len(results) >= int(max_results):
                        return results

    return results


@tool
def find_call_sites(func_name: str, max_results: int = 25) -> list[dict]:
    """
    Find AST-confirmed call sites of a function/method name (e.g. '_writeResults', 'shmoo').
    Matches by the final name only (attribute calls count).
    Returns: {file, lineno, called, snippet}
    """
    target = (func_name or "").strip()
    if not target:
        return []

    results: list[dict] = []

    for fp in iter_py_files(CODE_ROOT):
        try:
            src = _read_text(fp)
            lines = src.splitlines()
            tree = ast.parse(src)
        except Exception:
            continue

        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                called = _called_name(n)
                if called == target:
                    results.append({
                        "file": str(fp),
                        "lineno": getattr(n, "lineno", 0),
                        "called": called,
                        "snippet": _src_line(lines, getattr(n, "lineno", 0)),
                    })
                    if len(results) >= int(max_results):
                        return results

    return results


@tool
def extract_call_arguments(file_path: str, lineno: int, func_name: Optional[str] = None) -> dict:
    """
    Extract arguments from the Call expression on/near a given line in a file.
    If func_name is provided, prefer a call to that function name on that line.

    Returns:
      {
        "file": str,
        "lineno": int,
        "called": str,
        "args": [str, ...],
        "kwargs": [{"name": str, "value": str}, ...],
        "call_src": str
      }
    """
    p = Path(file_path)
    src = _read_text(p)
    lines = src.splitlines()

    line = max(1, int(lineno))
    # search window: exact line, then +/- 3 lines
    candidate_lines = [line, line - 1, line + 1, line - 2, line + 2, line - 3, line + 3]
    candidate_lines = [ln for ln in candidate_lines if 1 <= ln <= len(lines)]

    try:
        tree = ast.parse(src)
    except Exception:
        return {"error": f"Failed to parse file: {file_path}"}

    # group calls by lineno
    calls_by_line: dict[int, list[ast.Call]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and hasattr(n, "lineno"):
            calls_by_line.setdefault(n.lineno, []).append(n)

    target = (func_name or "").strip() if func_name else None

    def pick_call(calls: list[ast.Call]) -> Optional[ast.Call]:
        if not calls:
            return None
        if target:
            for c in calls:
                if _called_name(c) == target:
                    return c
        return calls[0]

    chosen_call: Optional[ast.Call] = None
    chosen_line: Optional[int] = None

    for ln in candidate_lines:
        calls = calls_by_line.get(ln, [])
        c = pick_call(calls)
        if c is not None:
            chosen_call = c
            chosen_line = ln
            break

    if chosen_call is None or chosen_line is None:
        return {
            "file": str(p),
            "lineno": line,
            "error": "No call expression found on/near that line.",
            "hint": "Try providing the exact call site line number.",
        }

    called = _called_name(chosen_call)

    def unparse(node: ast.AST) -> str:
        try:
            return ast.unparse(node)  # py3.9+
        except Exception:
            return "..."

    args = [unparse(a) for a in chosen_call.args]
    kwargs = []
    for kw in chosen_call.keywords:
        if kw.arg is None:
            kwargs.append({"name": "**", "value": unparse(kw.value)})
        else:
            kwargs.append({"name": kw.arg, "value": unparse(kw.value)})

    return {
        "file": str(p),
        "lineno": chosen_line,
        "called": called,
        "args": args,
        "kwargs": kwargs,
        "call_src": _src_line(lines, chosen_line),
    }


# ----------------------------
# Indexer (AST -> code_index.json)
# ----------------------------
@dataclass
class Symbol:
    kind: str                # "function" | "class"
    name: str
    qualname: str            # dotted path-ish: package.module:Class.method
    file: str
    lineno: int
    signature: str
    docstring: str


def func_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = fn.args
    parts: list[str] = []

    def fmt_arg(arg: ast.arg, default: Optional[ast.AST] = None) -> str:
        s = arg.arg
        if arg.annotation is not None:
            s += f": {safe_unparse(arg.annotation)}"
        if default is not None:
            s += f" = {safe_unparse(default)}"
        return s

    posonly = getattr(a, "posonlyargs", [])
    defaults = list(a.defaults)

    positional = list(posonly) + list(a.args)
    n_defaults = len(defaults)

    for i, arg in enumerate(positional):
        default = None
        if n_defaults and i >= len(positional) - n_defaults:
            default = defaults[i - (len(positional) - n_defaults)]
        parts.append(fmt_arg(arg, default))

    if posonly:
        parts.insert(len(posonly), "/")

    if a.vararg is not None:
        parts.append("*" + fmt_arg(a.vararg))

    if a.kwonlyargs:
        if a.vararg is None:
            parts.append("*")
        for arg, default in zip(a.kwonlyargs, a.kw_defaults):
            parts.append(fmt_arg(arg, default))

    if a.kwarg is not None:
        parts.append("**" + fmt_arg(a.kwarg))

    ret = ""
    if fn.returns is not None:
        ret = f" -> {safe_unparse(fn.returns)}"

    return f"({', '.join(parts)}){ret}"


def module_qualname(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


class SymbolVisitor(ast.NodeVisitor):
    def __init__(self, root: Path, file_path: Path):
        self.root = root
        self.file_path = file_path
        self.module = module_qualname(root, file_path)
        self.stack: list[str] = []
        self.symbols: list[Symbol] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        qual = f"{self.module}:{'.'.join(self.stack + [node.name])}"
        doc = ast.get_docstring(node) or ""
        self.symbols.append(Symbol(
            kind="class",
            name=node.name,
            qualname=qual,
            file=str(self.file_path),
            lineno=node.lineno,
            signature="",
            docstring=doc.strip(),
        ))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_fn(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_fn(node)
        self.generic_visit(node)

    def _handle_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        qual = f"{self.module}:{'.'.join(self.stack + [node.name])}"
        doc = ast.get_docstring(node) or ""
        sig = func_signature(node)
        self.symbols.append(Symbol(
            kind="function",
            name=node.name,
            qualname=qual,
            file=str(self.file_path),
            lineno=node.lineno,
            signature=sig,
            docstring=doc.strip(),
        ))


def index_repo(root: Path) -> list[Symbol]:
    out: list[Symbol] = []
    for fp in iter_py_files(root):
        try:
            src = _read_text(fp)
            tree = ast.parse(src)
            v = SymbolVisitor(root, fp)
            v.visit(tree)
            out.extend(v.symbols)
        except Exception as e:
            print(f"[warn] failed to parse {fp}: {e}")
    return out


def build_index(codebase_path: str) -> Path:
    root = Path(codebase_path).resolve()
    symbols = index_repo(root)
    payload = [asdict(s) for s in symbols]

    out_path = root / "code_index.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(symbols)} symbols to {out_path}")
    return out_path


if __name__ == "__main__":
    # If you run this file directly, it will build the index for your suite dir.
    codebase_path = r"path_to_src"
    build_index(codebase_path)
