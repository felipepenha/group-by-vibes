"""SQL AST canonicalization and syntactic tokenization wrapper."""

from __future__ import annotations

import logging

from sqlglot import exp
from sqlglot.errors import SqlglotError

from .dialects import normalize_dialect, validate_char_limit
from .parser import parse_sql, tokenize_sql

logger = logging.getLogger("group_by_vibes")

_LOGICAL_CLASSES = (exp.And, exp.Or, exp.Xor)


def _render_sql(
    node: exp.Expr,
    dialect: str,
    strip_comments: bool,
    include_semicolons: bool = False,
) -> str:
    rendered = (
        node.sql(
            dialect=dialect,
            normalize=True,
            pretty=False,
            comments=not strip_comments,
        )
        .strip()
        .rstrip(";")
        .strip()
    )
    if include_semicolons and rendered:
        return f"{rendered};"
    return rendered


def _flatten_assoc(node: exp.Binary, cls: type[exp.Binary]) -> list[exp.Expr]:
    """Flatten nested associative boolean nodes while keeping mixed-operator parens."""
    flattened: list[exp.Expr] = []
    for child in (node.this, node.expression):
        if child is None:
            continue
        cur = child
        while isinstance(cur, exp.Paren) and cur.this is not None:
            inner = cur.this
            if isinstance(inner, cls) or not isinstance(inner, _LOGICAL_CLASSES):
                cur = inner
            else:
                break
        if isinstance(cur, cls):
            flattened.extend(_flatten_assoc(cur, cls))
        else:
            flattened.append(cur.copy())
    return flattened


def _is_inside_same_assoc(node: exp.Expr, cls: type[exp.Binary]) -> bool:
    parent = node.parent
    while isinstance(parent, exp.Paren):
        parent = parent.parent
    return isinstance(parent, cls)


def _rewrite_commutative(ast: exp.Expr, dialect: str, strip_comments: bool) -> exp.Expr:
    """Sort commutative predicates bottom-up (post-order) and return updated root."""
    for key, val in list(ast.args.items()):
        if isinstance(val, exp.Expr):
            new_val = _rewrite_commutative(val, dialect, strip_comments)
            if new_val is not val:
                ast.set(key, new_val)
        elif isinstance(val, list):
            changed = False
            new_list: list[object] = []
            for item in val:
                if isinstance(item, exp.Expr):
                    new_item = _rewrite_commutative(item, dialect, strip_comments)
                    changed = changed or (new_item is not item)
                    new_list.append(new_item)
                else:
                    new_list.append(item)
            if changed:
                ast.set(key, new_list)

    if isinstance(ast, (exp.EQ, exp.NEQ)):
        if ast.this is not None and ast.expression is not None:
            left_sql = _render_sql(ast.this, dialect, strip_comments)
            right_sql = _render_sql(ast.expression, dialect, strip_comments)
            if left_sql > right_sql:
                left_copy = ast.this.copy()
                right_copy = ast.expression.copy()
                ast.set("this", right_copy)
                ast.set("expression", left_copy)
        return ast

    if isinstance(ast, exp.In):
        if ast.args.get("query") is None and len(ast.expressions) > 1:
            sorted_exprs = sorted(
                ast.expressions,
                key=lambda expr: _render_sql(expr, dialect, strip_comments),
            )
            ast.set("expressions", [expr.copy() for expr in sorted_exprs])
        return ast

    if isinstance(ast, (exp.And, exp.Or)):
        cls = type(ast)
        if _is_inside_same_assoc(ast, cls):
            return ast
        args = _flatten_assoc(ast, cls)
        if len(args) > 1:
            args.sort(key=lambda expr: _render_sql(expr, dialect, strip_comments))
            rebuilt: exp.Expr = args[0].copy()
            for arg in args[1:]:
                rebuilt = cls(this=rebuilt, expression=arg.copy())
            return rebuilt
    return ast


def canonicalize_ast(
    ast: exp.Expr,
    *,
    dialect: str = "postgres",
    sort_commutative: bool = True,
    normalize_identifiers: bool = True,
    strip_comments: bool = True,
    include_semicolons: bool = False,
) -> tuple[str, bool]:
    """Canonicalize an already-parsed AST without re-parsing raw text."""
    try:
        work = ast.copy()
        if normalize_identifiers:
            for ident in work.find_all(exp.Identifier):
                if not ident.quoted and isinstance(ident.this, str):
                    ident.set("this", ident.this.lower())

        if sort_commutative:
            work = _rewrite_commutative(work, dialect, strip_comments)

        return (
            _render_sql(
                work,
                dialect,
                strip_comments=strip_comments,
                include_semicolons=include_semicolons,
            ),
            True,
        )
    except (SqlglotError, RecursionError) as exc:
        logger.debug("Canonicalization failed (dialect=%r, error=%s)", dialect, type(exc).__name__)
        return "", False


def try_canonicalize_sql(
    query: str,
    *,
    dialect: str = "postgres",
    sort_commutative: bool = True,
    normalize_identifiers: bool = True,
    strip_comments: bool = True,
    include_semicolons: bool = False,
    max_query_chars: int | None = 65_536,
) -> tuple[str, bool]:
    """Parse and canonicalize `query`, returning `(canonical_sql, parse_succeeded)`."""
    norm_dialect = normalize_dialect(dialect)
    validate_char_limit("max_query_chars", max_query_chars)
    if not isinstance(query, str):
        return "", False

    parsed = parse_sql(query, dialect=norm_dialect, max_query_chars=max_query_chars)
    if not parsed.ok or parsed.ast is None:
        return "", False

    rendered, ok = canonicalize_ast(
        parsed.ast,
        dialect=norm_dialect,
        sort_commutative=sort_commutative,
        normalize_identifiers=normalize_identifiers,
        strip_comments=strip_comments,
        include_semicolons=include_semicolons,
    )
    if not ok:
        return "", False
    return rendered, True


def canonicalize_sql(
    query: str,
    dialect: str = "postgres",
    *,
    sort_commutative: bool = True,
    normalize_identifiers: bool = True,
    strip_comments: bool = True,
    include_semicolons: bool = False,
    max_query_chars: int | None = 65_536,
) -> str:
    """Return canonicalized SQL string (or normalized fallback text on syntax error)."""
    canon, _ = try_canonicalize_sql(
        query,
        dialect=dialect,
        sort_commutative=sort_commutative,
        normalize_identifiers=normalize_identifiers,
        strip_comments=strip_comments,
        include_semicolons=include_semicolons,
        max_query_chars=max_query_chars,
    )
    return canon


def sql_tokenizer(
    text: str,
    *,
    dialect: str = "postgres",
    include_semicolons: bool = False,
    strip_comments: bool = True,
    max_query_chars: int | None = 65_536,
) -> list[str]:
    """Tokenize SQL into syntactic tokens via `parser.tokenize_sql`."""
    return tokenize_sql(
        text,
        dialect=dialect,
        include_semicolons=include_semicolons,
        strip_comments=strip_comments,
        max_query_chars=max_query_chars,
    )
