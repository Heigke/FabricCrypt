"""z351 — Clean-card preprocessor for Sebas's M1/M2 130nm BSIM4 cards.

The cards use HSPICE-style expressions inside .model bodies that ngspice-42
silently drops (e.g. `rdsw = 100-140*1e6*1u/int(1u/0.34u)`, `cgso = rcgon*3.65e-10`).
The .param macros (rcgon, rcjn, toxn, vth0n, lpe0n, vsatn, lintn, wintn, k3n,
pvth0n, rcjswn, rcjswgn) live in M2 only; M1 relies on cross-file scope when
both files are .include'd together.

This preprocessor:
  1. Collects all .param defs from BOTH cards (M2 first, then M1 — deck order)
  2. Walks the .model body and substitutes every RHS using a sympy-grade
     expression evaluator that understands:
       - SPICE eng suffixes (1u → 1e-6, 4n → 4e-9, ...)
       - int(x), abs(x), max, min
       - * / + - ( )
       - identifier lookup (rcgon → 1, toxn → 4e-9, ...)
  3. Writes M1_130DNWFB_CLEAN.txt and M2_130bulkNSRAM_CLEAN.txt with all
     RHS values as literal Python floats in scientific notation.

Usage (CLI):
    python scripts/clean_card.py
        --m1 data/sebas_2026_04_22/M1_130DNWFB.txt
        --m2 data/sebas_2026_04_22/M2_130bulkNSRAM.txt
        --out results/z351_clean_card/
"""
from __future__ import annotations
import argparse, math, re
from pathlib import Path


# -- SPICE suffix table (ngspice + HSPICE compatible) -----------------------
SUFFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "mil": 25.4e-6,
    "k": 1e3, "meg": 1e6, "x": 1e6, "g": 1e9, "t": 1e12,
}
# 'm' is ambiguous (mili-vs-meg). HSPICE: m=mili. SPICE3/ngspice: m=mili.
SUFFIX["m"] = 1e-3


_NUM_RE = re.compile(
    r"""^([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)([a-zA-Z]*)$"""
)


def parse_spice_number(token: str) -> float | None:
    """Return literal float for SPICE numeric token, else None."""
    t = token.strip()
    try:
        return float(t)
    except ValueError:
        pass
    m = _NUM_RE.match(t)
    if not m:
        return None
    try:
        base = float(m.group(1))
    except ValueError:
        return None
    suf = m.group(2).lower()
    if not suf:
        return base
    if suf in SUFFIX:
        return base * SUFFIX[suf]
    # Try 3-letter, then first letter
    if suf[:3] in SUFFIX:
        return base * SUFFIX[suf[:3]]
    if suf[0] in SUFFIX:
        return base * SUFFIX[suf[0]]
    return None


# -- expression evaluator ---------------------------------------------------
class _ExprEval:
    """Recursive-descent eval over SPICE/HSPICE expressions.

    Grammar:
        expr   := term (('+'|'-') term)*
        term   := factor (('*'|'/') factor)*
        factor := ('+'|'-') factor | atom ('**' factor)?
        atom   := NUMBER | IDENT | IDENT '(' arglist ')' | '(' expr ')'
    """

    FUNCS = {
        "int":  lambda x: float(int(x)),
        "abs":  abs,
        "sqrt": math.sqrt,
        "exp":  math.exp,
        "log":  math.log,
        "log10": math.log10,
        "max":  max,
        "min":  min,
        "pow":  pow,
    }

    def __init__(self, scope: dict[str, float]):
        self.scope = {k.lower(): v for k, v in scope.items()}
        self.tokens: list[str] = []
        self.pos = 0

    # -- tokenizer
    # Do NOT match unary +/- in the number rule; let _factor() handle signs.
    # Otherwise '100-140*1u' tokenizes as ['100', '-140', '*', '1u'] and the
    # parser sees an unexpected '-140' after '100'.
    _TOK_RE = re.compile(
        r"\s*(?:"
        r"(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?[a-zA-Z]*)"  # unsigned number with optional suffix
        r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
        r"|(?P<op>\*\*|<<|>>|==|!=|<=|>=|[+\-*/(),<>])"
        r")"
    )

    def _tokenize(self, src: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(src):
            m = self._TOK_RE.match(src, i)
            if not m or m.end() == i:
                # Skip whitespace
                if src[i].isspace():
                    i += 1
                    continue
                raise ValueError(f"bad char at {i}: {src[i]!r} in {src!r}")
            for k in ("num", "ident", "op"):
                v = m.group(k)
                if v is not None:
                    out.append(v)
                    break
            i = m.end()
        return out

    def parse(self, src: str) -> float:
        # Handle leading sign on the very first token: numbers like '-1.6e-8'
        # already parse correctly via _NUM_RE, but unary '-' on identifiers
        # is the factor rule's job.
        self.tokens = self._tokenize(src)
        self.pos = 0
        v = self._expr()
        if self.pos != len(self.tokens):
            raise ValueError(
                f"unconsumed input {self.tokens[self.pos:]!r} in {src!r}"
            )
        return v

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _eat(self, tok: str | None = None) -> str:
        t = self.tokens[self.pos]
        if tok is not None and t != tok:
            raise ValueError(f"expected {tok!r}, got {t!r}")
        self.pos += 1
        return t

    def _expr(self) -> float:
        v = self._term()
        while True:
            t = self._peek()
            if t == "+":
                self._eat()
                v = v + self._term()
            elif t == "-":
                self._eat()
                v = v - self._term()
            else:
                return v

    def _term(self) -> float:
        v = self._factor()
        while True:
            t = self._peek()
            if t == "*":
                self._eat()
                v = v * self._factor()
            elif t == "/":
                self._eat()
                d = self._factor()
                v = v / d if d != 0 else float("inf")
            else:
                return v

    def _factor(self) -> float:
        t = self._peek()
        if t == "+":
            self._eat()
            return self._factor()
        if t == "-":
            self._eat()
            return -self._factor()
        v = self._atom()
        if self._peek() == "**":
            self._eat()
            v = v ** self._factor()
        return v

    def _atom(self) -> float:
        t = self._eat()
        if t == "(":
            v = self._expr()
            self._eat(")")
            return v
        # number?
        n = parse_spice_number(t)
        if n is not None:
            return n
        # identifier — possibly a func call
        if self._peek() == "(":
            self._eat("(")
            args: list[float] = []
            if self._peek() != ")":
                args.append(self._expr())
                while self._peek() == ",":
                    self._eat(",")
                    args.append(self._expr())
            self._eat(")")
            fn = self.FUNCS.get(t.lower())
            if fn is None:
                raise ValueError(f"unknown function {t!r}")
            return float(fn(*args))
        # variable lookup
        key = t.lower()
        if key in self.scope:
            return self.scope[key]
        raise ValueError(f"undefined identifier {t!r}")


def eval_expr(src: str, scope: dict[str, float]) -> float:
    return _ExprEval(scope).parse(src)


# -- card walker ------------------------------------------------------------
PARAM_HEADER_RE = re.compile(r"^\s*\.param\b", re.IGNORECASE)
MODEL_HEADER_RE = re.compile(r"^\s*\.model\s+\S+\s+\w+", re.IGNORECASE)
CONT_RE          = re.compile(r"^\s*\+")
COMMENT_RE       = re.compile(r"^\s*\*")


def _join_continuations(lines: list[str]) -> list[tuple[int, str]]:
    """Return list of (start_line_idx, joined_text) for every logical SPICE
    statement (.param, .model, regular). Continuation lines starting with '+'
    are folded into the previous logical line."""
    merged: list[tuple[int, str]] = []
    for i, raw in enumerate(lines):
        if COMMENT_RE.match(raw):
            continue
        if CONT_RE.match(raw):
            if merged:
                start, prev = merged[-1]
                merged[-1] = (start, prev + " " + raw.lstrip()[1:])
            continue
        merged.append((i, raw))
    return merged


def _extract_param_scope(text: str, seed: dict[str, float] | None = None) -> dict[str, float]:
    """Walk .param blocks and return {name → numeric value}."""
    scope = dict(seed or {})
    merged = _join_continuations(text.splitlines())
    for _, line in merged:
        if not PARAM_HEADER_RE.match(line):
            continue
        body = re.sub(r"^\s*\.param\s+", "", line, flags=re.IGNORECASE)
        # Pair RHS by name = <token-or-expr-until-next-name=>
        # Strategy: split on '=' and reconstruct.
        # Easier: regex for `name = <stuff>` where stuff is everything up to
        # the next bareword followed by '='.
        # We accept that .param RHS in these cards are simple literals.
        for nm, rhs in _iter_assignments(body):
            try:
                v = eval_expr(rhs, scope)
                scope[nm.lower()] = v
            except Exception:
                pass
    return scope


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _iter_assignments(line: str):
    """Yield (name, rhs_expression) pairs from a single logical line.

    RHS extends from after '=' up to (but not including) the next `<name> =`
    or end of line.
    """
    # Find every '<name> =' boundary
    positions = []
    for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9]*)\s*=", line):
        # skip occurrences inside parentheses (function calls like int(1u/0.34u))
        # We track parens depth at this position
        depth = 0
        for ch in line[:m.start()]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        if depth == 0:
            positions.append((m.start(), m.end(), m.group(1)))
    for k, (st, en, name) in enumerate(positions):
        rhs_start = en
        rhs_end = positions[k + 1][0] if k + 1 < len(positions) else len(line)
        rhs = line[rhs_start:rhs_end].strip()
        if rhs:
            yield name, rhs


def clean_card(text: str, scope: dict[str, float]) -> tuple[str, dict]:
    """Substitute all RHS in .model body with literal floats.

    Returns (cleaned_text, report)
        report: {
            "expressions_evaluated": [(name, original_rhs, value), ...],
            "expressions_failed":    [(name, original_rhs, error), ...],
        }
    """
    lines = text.splitlines()
    merged = _join_continuations(lines)
    out_lines = list(lines)  # mutable copy

    report = {
        "expressions_evaluated": [],
        "expressions_failed": [],
        "scope_size": len(scope),
    }

    in_model = False
    for start_idx, joined in merged:
        if PARAM_HEADER_RE.match(joined):
            in_model = False
            continue
        if MODEL_HEADER_RE.match(joined):
            in_model = True
            # fall through: rewrite assignments on the same line (rare here)
        elif not in_model:
            continue
        # This is a model-body line or the .model header line itself.
        # Rebuild logical text but ONLY substitute on identifier-RHS,
        # leave pure-number RHS unchanged.
        new_body = _rewrite_assignments(joined, scope, report)
        # Now write the rewritten content back into the FIRST line; clear
        # continuation lines (they were folded). But preserving multi-line
        # layout is impossible after substitution → emit single line.
        # Find continuation span:
        j = start_idx + 1
        while j < len(lines) and CONT_RE.match(lines[j]):
            out_lines[j] = ""  # blank out
            j += 1
        out_lines[start_idx] = new_body

    # Drop blanks introduced by folding (preserve other blank lines)
    final = "\n".join(ln for ln in out_lines)
    # collapse runs of blank lines to single blank for tidiness
    final = re.sub(r"\n{3,}", "\n\n", final)
    return final, report


def _rewrite_assignments(line: str, scope: dict[str, float], report: dict) -> str:
    """Inside a logical line, replace every `name = <expr>` with
    `name = <literal>` if <expr> evaluates."""
    # Preserve a header prefix if .model line
    m_model = MODEL_HEADER_RE.match(line)
    out_chunks: list[str] = []
    cursor = 0
    if m_model:
        # Keep the ".model NAME TYPE" prefix verbatim
        out_chunks.append(line[: m_model.end()])
        cursor = m_model.end()

    # collect assignment boundaries on the rest
    rest = line[cursor:]
    positions = []
    for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9]*)\s*=", rest):
        depth = 0
        for ch in rest[:m.start()]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        if depth == 0:
            positions.append((m.start(), m.end(), m.group(1)))

    if not positions:
        return line

    # text before first assignment (e.g. leading '+')
    out_chunks.append(rest[: positions[0][0]])
    # Strip any stray '+' from continuation; we always write a fresh '+'
    out_chunks[-1] = out_chunks[-1].replace("+", " ")

    is_first = (m_model is not None)
    for k, (st, en, name) in enumerate(positions):
        rhs_start = en
        rhs_end = positions[k + 1][0] if k + 1 < len(positions) else len(rest)
        raw_rhs = rest[rhs_start:rhs_end].strip()
        try:
            v = eval_expr(raw_rhs, scope)
            new_rhs = f"{v:.10g}"
            report["expressions_evaluated"].append((name, raw_rhs, v))
        except Exception as e:
            new_rhs = raw_rhs
            report["expressions_failed"].append((name, raw_rhs, str(e)))

        # First assignment after .model header gets '+ ' prefix; subsequent
        # ones get a space separator.
        if is_first and m_model is not None:
            out_chunks.append("\n+ ")
            is_first = False
        elif k > 0:
            out_chunks.append("  ")
        else:
            # On a continuation line (no .model header), preserve a leading +
            # by writing '+ ' once at start of body.
            if not m_model:
                out_chunks.append("+ ")
        out_chunks.append(f"{name} = {new_rhs}")
    return "".join(out_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default="data/sebas_2026_04_22/M1_130DNWFB.txt")
    ap.add_argument("--m2", default="data/sebas_2026_04_22/M2_130bulkNSRAM.txt")
    ap.add_argument("--out", default="results/z351_clean_card/")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    m1_path = (root / args.m1).resolve()
    m2_path = (root / args.m2).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    m1_raw = m1_path.read_text()
    m2_raw = m2_path.read_text()

    # M2 holds the .param block; M1 references those names. Build joint scope.
    scope = _extract_param_scope(m2_raw)
    scope = _extract_param_scope(m1_raw, seed=scope)  # M1 may add more

    print(f"[scope] {len(scope)} params:")
    for k in sorted(scope):
        print(f"  {k:12s} = {scope[k]}")

    m1_clean, m1_rep = clean_card(m1_raw, scope)
    m2_clean, m2_rep = clean_card(m2_raw, scope)

    (out_dir / "M1_130DNWFB_CLEAN.txt").write_text(m1_clean)
    (out_dir / "M2_130bulkNSRAM_CLEAN.txt").write_text(m2_clean)

    import json
    rep = {
        "scope": {k: scope[k] for k in sorted(scope)},
        "M1": {
            "evaluated_count": len(m1_rep["expressions_evaluated"]),
            "failed_count":    len(m1_rep["expressions_failed"]),
            "top_expressions": [
                {"name": n, "rhs": r, "value": v}
                for n, r, v in m1_rep["expressions_evaluated"]
                if not re.match(r"^[+-]?\d", r)  # non-trivial (not pure number)
            ][:50],
            "failed": [
                {"name": n, "rhs": r, "err": e}
                for n, r, e in m1_rep["expressions_failed"]
            ][:50],
        },
        "M2": {
            "evaluated_count": len(m2_rep["expressions_evaluated"]),
            "failed_count":    len(m2_rep["expressions_failed"]),
            "top_expressions": [
                {"name": n, "rhs": r, "value": v}
                for n, r, v in m2_rep["expressions_evaluated"]
                if not re.match(r"^[+-]?\d", r)
            ][:50],
            "failed": [
                {"name": n, "rhs": r, "err": e}
                for n, r, e in m2_rep["expressions_failed"]
            ][:50],
        },
    }
    (out_dir / "clean_card_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\n[ok] wrote {out_dir/'M1_130DNWFB_CLEAN.txt'}")
    print(f"[ok] wrote {out_dir/'M2_130bulkNSRAM_CLEAN.txt'}")
    print(f"[ok] wrote {out_dir/'clean_card_report.json'}")


if __name__ == "__main__":
    main()
