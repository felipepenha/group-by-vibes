"""Shared dialect-general SQL AST normalization, clause decomposition, and matching engine."""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias, TypeVar

from sqlglot import exp
from sqlglot.errors import SqlglotError

from .base import BaseEvaluator
from .dialects import (
    normalize_dialect,
    normalize_grounding_identifier,
    split_qualified_table_parts,
)
from .extract import CompiledExtractor, ExtractorLike, apply_extractor
from .grounding import (
    _in_clause_for_select,
    _NormalizedSchema,
    _select_alias_names,
    normalize_schema,
)
from .parser import ParsedSql, parse_sql

COMPONENT_TYPES: tuple[str, ...] = (
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
    "cte",
    "join_on",
)
HARDNESS_LEVELS: tuple[str, ...] = ("easy", "medium", "hard", "extra")
ALL_HARDNESS_LEVELS: tuple[str, ...] = (*HARDNESS_LEVELS, "all")

SchemaDict: TypeAlias = Mapping[str, Iterable[str]]
ForeignKeyPair: TypeAlias = tuple[str, str]
ForeignKeysLike: TypeAlias = Sequence[ForeignKeyPair]
_CanonIdent: TypeAlias = tuple[str, bool]
_CanonTableParts: TypeAlias = tuple[_CanonIdent | None, _CanonIdent | None, _CanonIdent]
_ForeignKeyMap: TypeAlias = dict[tuple[str, str], tuple[_CanonIdent, _CanonIdent]]
_T = TypeVar("_T")

_SQLITE_IMPLICIT_COLUMNS = frozenset({"rowid", "_rowid_", "oid"})
_LOGICAL_CLASSES = (exp.And, exp.Or, exp.Xor)
_SUBQUERY_BOUNDARY_CLASSES = (exp.Select, exp.SetOperation, exp.Subquery, exp.CTE)
_AGG_STRIP_CLASSES = (exp.Max, exp.Min, exp.Count, exp.Sum, exp.Avg)
_WHERE_TARGET_OP_CLASSES = (
    exp.EQ,
    exp.NEQ,
    exp.LT,
    exp.LTE,
    exp.GT,
    exp.GTE,
    exp.Like,
    exp.ILike,
    exp.Is,
    exp.Between,
    exp.In,
)
_INEQUALITY_FLIP: dict[type[exp.Binary], type[exp.Binary]] = {
    exp.LT: exp.GT,
    exp.LTE: exp.GTE,
    exp.GT: exp.LT,
    exp.GTE: exp.LTE,
}
_MODIFIER_KEYS = ("order", "limit", "offset")


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration flags governing AST normalization and matching strictness."""

    dialect: str
    ignore_values: bool = True
    ignore_distinct: bool = True
    include_limit_value: bool = False
    include_join_conditions: bool = False
    skip_unparseable_predictions: bool = False


@dataclass(frozen=True)
class ComponentScore:
    """Per-example exact-set component match result (`precision`, `recall`, `f1` in `{0, 1}`)."""

    precision: int
    recall: int
    f1: int
    label_total: int
    pred_total: int


@dataclass(frozen=True)
class NormalizedQuery:
    """Canonicalized, dialect-general decomposition of a SQL query or subquery."""

    is_empty: bool
    select_distinct: bool
    select_distinct_on: str
    select_items: tuple[str, ...]
    select_no_agg: tuple[str, ...]
    from_tables: tuple[str, ...]
    from_subqueries: tuple[NormalizedQuery, ...]
    join_on: tuple[str, ...]
    where_preds: tuple[str, ...]
    where_no_op: tuple[str, ...]
    where_and_or: frozenset[str]
    group_by_no_table: tuple[str, ...]
    group_by_full: tuple[str, ...]
    having_preds: tuple[str, ...]
    having_and_or: frozenset[str]
    order_by: tuple[tuple[str, str, str], ...]
    has_limit: bool
    limit_value: str | None
    iuen_branches: tuple[tuple[str, NormalizedQuery], ...]
    ctes: tuple[tuple[tuple[str, ...], NormalizedQuery], ...]
    keywords: frozenset[str]
    hardness: str
    canonical_signature: str


@dataclass(frozen=True)
class PairMatchResult:
    """Outcome of scoring one prediction decomposition against one winning reference."""

    exact_match: int
    hardness: str
    components: dict[str, ComponentScore]
    prediction_parsed: bool
    schema_applied: bool
    foreign_keys_applied: bool


class _AliasTarget(NamedTuple):
    kind: str  # "physical" | "cte" | "derived" | "opaque"
    canon_parts: _CanonTableParts | None
    norm_tbl: str | None
    schema_cols: frozenset[str] | None
    placeholder: str | None


@dataclass(frozen=True)
class _CteSpec:
    pos: int
    raw_norm: str
    col_aliases: tuple[str, ...]
    body_ast: exp.Expr


def _compute_canonical_signature(
    *,
    is_empty: bool,
    select_distinct: bool,
    select_distinct_on: str,
    select_items: tuple[str, ...],
    from_tables: tuple[str, ...],
    from_subqueries: tuple[NormalizedQuery, ...],
    join_on: tuple[str, ...],
    where_preds: tuple[str, ...],
    where_and_or: frozenset[str],
    group_by_full: tuple[str, ...],
    having_preds: tuple[str, ...],
    having_and_or: frozenset[str],
    order_by: tuple[tuple[str, str, str], ...],
    has_limit: bool,
    limit_value: str | None,
    iuen_branches: tuple[tuple[str, NormalizedQuery], ...],
    ctes: tuple[tuple[tuple[str, ...], NormalizedQuery], ...],
    keywords: frozenset[str],
) -> str:
    payload = {
        "is_empty": is_empty,
        "select_distinct": select_distinct,
        "select_distinct_on": select_distinct_on,
        "select_items": sorted(select_items),
        "from_tables": sorted(from_tables),
        "from_subqueries": sorted(sq.canonical_signature for sq in from_subqueries),
        "join_on": sorted(join_on),
        "where_preds": sorted(where_preds),
        "where_and_or": sorted(where_and_or),
        "group_by_full": sorted(group_by_full),
        "having_preds": sorted(having_preds),
        "having_and_or": sorted(having_and_or),
        "order_by": [list(item) for item in order_by],
        "has_limit": has_limit,
        "limit_value": limit_value,
        "iuen_branches": sorted([op, branch.canonical_signature] for op, branch in iuen_branches),
        "ctes": sorted([list(cols), body.canonical_signature] for cols, body in ctes),
        "keywords": sorted(keywords),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _render_expr(node: exp.Expr, dialect: str) -> str:
    return (
        node.sql(dialect=dialect, normalize=True, pretty=False, comments=False)
        .strip()
        .rstrip(";")
        .strip()
    )


def _canon_ident(name: str, quoted: bool) -> _CanonIdent:
    return (name if quoted else name.lower(), bool(quoted))


def _make_ident(canon: _CanonIdent) -> exp.Identifier:
    return exp.Identifier(this=canon[0], quoted=canon[1])


def coerce_reference_group(reference: Any, index: int | None = None) -> list[str]:
    """Validate and coerce a reference SQL string or non-empty reference sequence."""
    loc = f" at index {index}" if index is not None else ""
    if isinstance(reference, str):
        return [reference]
    if isinstance(reference, (bytes, bytearray, Mapping)) or not isinstance(reference, Iterable):
        raise TypeError(
            f"Expected str or Sequence[str] reference{loc}, got {type(reference).__name__}."
        )
    refs = list(reference)
    if not refs:
        raise ValueError(f"Reference group{loc} must not be empty.")
    for item in refs:
        if not isinstance(item, str):
            raise TypeError(f"Reference entries{loc} must be str, got {type(item).__name__}.")
    return refs


def normalize_foreign_keys(
    foreign_keys: ForeignKeysLike | None,
    dialect: str = "postgres",
) -> _ForeignKeyMap:
    """Validate per-row FK column pairs and build a transitive union-find equivalence map."""
    if foreign_keys is None:
        return {}
    if isinstance(foreign_keys, (str, bytes, bytearray, Mapping)) or not isinstance(
        foreign_keys, Sequence
    ):
        raise TypeError(
            "`foreign_keys` must be a non-str/non-mapping Sequence of (left, right) pairs or None, "
            f"got {type(foreign_keys).__name__}."
        )
    norm_dialect = normalize_dialect(dialect)
    if len(foreign_keys) == 0:
        return {}

    parsed_pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    canon_by_key: dict[tuple[str, str], tuple[_CanonIdent, _CanonIdent]] = {}

    for idx, entry in enumerate(foreign_keys):
        if isinstance(entry, (str, bytes, bytearray, Mapping)) or not isinstance(entry, Sequence):
            raise TypeError(
                f"`foreign_keys[{idx}]` must be a 2-element sequence of str, "
                f"got {type(entry).__name__}."
            )
        if len(entry) != 2:
            raise ValueError(
                f"`foreign_keys[{idx}]` must contain exactly 2 items, got {len(entry)}."
            )
        left_raw, right_raw = entry[0], entry[1]
        if not isinstance(left_raw, str) or not isinstance(right_raw, str):
            raise TypeError(
                f"`foreign_keys[{idx}]` entries must be str, "
                f"got ({type(left_raw).__name__}, {type(right_raw).__name__})."
            )
        if not left_raw.strip() or not right_raw.strip():
            raise ValueError(f"`foreign_keys[{idx}]` endpoints must be non-empty strings.")

        endpoints: list[tuple[str, str]] = []
        for endpoint in (left_raw, right_raw):
            cat_part, tbl_part, col_part = split_qualified_table_parts(
                endpoint, dialect=norm_dialect
            )
            if cat_part is not None or tbl_part is None:
                raise ValueError(
                    f"Foreign key endpoint {endpoint!r} must be strictly 2-part `'table.column'`."
                )
            t_name, t_quoted = tbl_part
            c_name, c_quoted = col_part
            lookup_key = (
                normalize_grounding_identifier(t_name, t_quoted, norm_dialect, is_table=True),
                normalize_grounding_identifier(c_name, c_quoted, norm_dialect, is_table=False),
            )
            canon_val = (_canon_ident(t_name, t_quoted), _canon_ident(c_name, c_quoted))
            existing = canon_by_key.get(lookup_key)
            if existing is None or canon_val < existing:
                canon_by_key[lookup_key] = canon_val
            endpoints.append(lookup_key)
        parsed_pairs.append((endpoints[0], endpoints[1]))

    parent: dict[tuple[str, str], tuple[str, str]] = {key: key for key in canon_by_key}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left_key, right_key in parsed_pairs:
        ra, rb = find(left_key), find(right_key)
        if ra != rb:
            parent[rb] = ra

    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in canon_by_key:
        groups[find(key)].append(key)

    kmap: _ForeignKeyMap = {}
    for members in groups.values():
        rep = min(canon_by_key[m] for m in members)
        for member in members:
            kmap[member] = rep
    return kmap


def _merge_with_clauses(outer_with: exp.With, inner_with: exp.With) -> exp.With:
    merged_exprs = [e.copy() for e in outer_with.expressions] + [
        e.copy() for e in inner_with.expressions
    ]
    recursive = bool(outer_with.args.get("recursive")) or bool(inner_with.args.get("recursive"))
    return exp.With(expressions=merged_exprs, recursive=recursive)


def _unwrap_query_subquery(ast: exp.Expr) -> exp.Expr:
    """Unwrap outer `exp.Subquery` parens on a fresh copy while hoisting query modifiers."""
    cur = ast.copy()
    while isinstance(cur, exp.Subquery):
        inner = cur.this
        if inner is None:
            break
        inner_copy = inner.copy()
        outer_with = cur.args.get("with_")
        if isinstance(outer_with, exp.With):
            inner_with = inner_copy.args.get("with_")
            if isinstance(inner_with, exp.With):
                inner_copy.set("with_", _merge_with_clauses(outer_with, inner_with))
            else:
                inner_copy.set("with_", outer_with.copy())
        for mod_key in _MODIFIER_KEYS:
            outer_mod = cur.args.get(mod_key)
            if outer_mod is not None:
                inner_copy.set(mod_key, outer_mod.copy())
        cur = inner_copy
    return cur


def parse_query_ast(
    raw_sql: Any,
    *,
    dialect: str,
    max_query_chars: int | None,
    max_input_chars: int | None,
    compiled_extractor: CompiledExtractor,
    apply_extract: bool,
) -> ParsedSql:
    """Extract (optional), parse, unwrap parens, and verify `raw_sql` is a SQL query AST."""
    if apply_extract and compiled_extractor is not None:
        ext = apply_extractor(raw_sql, compiled_extractor, max_input_chars)
        if not ext.ok or ext.sql is None:
            return ParsedSql(
                ast=None,
                ok=False,
                failure_reason=ext.failure_reason or "extraction_failed",
            )
        sql_text = ext.sql
    elif not isinstance(raw_sql, str):
        return ParsedSql(ast=None, ok=False, failure_reason="syntax_error")
    else:
        sql_text = raw_sql

    parsed = parse_sql(sql_text, dialect=dialect, max_query_chars=max_query_chars)
    if not parsed.ok or parsed.ast is None:
        return parsed

    try:
        root = _unwrap_query_subquery(parsed.ast)
    except (SqlglotError, RecursionError):
        return ParsedSql(ast=None, ok=False, failure_reason="syntax_error")

    if not isinstance(root, (exp.Select, exp.SetOperation)):
        return ParsedSql(ast=None, ok=False, failure_reason="non_query")
    return ParsedSql(ast=root, ok=True, failure_reason=None)


def _walk_current_level(
    root: exp.Expr,
    *,
    skip_from_and_join_targets: bool = True,
) -> Iterable[exp.Expr]:
    """Yield `root` and descendants at the current query level without entering subqueries."""
    stack: list[exp.Expr] = [root]
    while stack:
        node = stack.pop()
        yield node
        for key, val in node.args.items():
            if node is root and skip_from_and_join_targets and key == "from_":
                continue
            if node is root and key == "with_":
                continue
            if isinstance(node, exp.Join) and skip_from_and_join_targets and key == "this":
                continue
            if isinstance(val, exp.Expr):
                if not isinstance(val, _SUBQUERY_BOUNDARY_CLASSES):
                    stack.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, exp.Expr) and not isinstance(
                        item, _SUBQUERY_BOUNDARY_CLASSES
                    ):
                        stack.append(item)


def _fold_ident_inplace(ident: exp.Identifier) -> None:
    if not ident.quoted and isinstance(ident.this, str):
        ident.set("this", ident.this.lower())


def _extract_cte_specs(with_node: exp.With, dialect: str) -> list[_CteSpec]:
    specs: list[_CteSpec] = []
    for idx, cte_node in enumerate(with_node.expressions):
        if not isinstance(cte_node, exp.CTE) or cte_node.this is None:
            continue
        alias_node = cte_node.args.get("alias")
        name_ident: exp.Identifier | None = None
        col_aliases: tuple[str, ...] = ()
        if isinstance(alias_node, exp.TableAlias):
            if isinstance(alias_node.this, exp.Identifier):
                name_ident = alias_node.this
            folded_cols: list[str] = []
            for col_expr in alias_node.columns:
                col_copy = col_expr.copy()
                if isinstance(col_copy, exp.Identifier):
                    _fold_ident_inplace(col_copy)
                else:
                    for sub_ident in col_copy.find_all(exp.Identifier):
                        _fold_ident_inplace(sub_ident)
                folded_cols.append(_render_expr(col_copy, dialect))
            col_aliases = tuple(folded_cols)
        elif isinstance(alias_node, exp.Identifier):
            name_ident = alias_node
        if name_ident is None or not name_ident.name:
            continue
        raw_norm = normalize_grounding_identifier(
            name_ident.name, bool(name_ident.quoted), dialect, is_table=True
        )
        specs.append(
            _CteSpec(
                pos=idx,
                raw_norm=raw_norm,
                col_aliases=col_aliases,
                body_ast=cte_node.this.copy(),
            )
        )
    return specs


def _process_with_clause(
    with_node: exp.With,
    *,
    dialect: str,
    config: MatchingConfig,
    norm_schema: _NormalizedSchema | None,
    kmap: _ForeignKeyMap,
    active_cte_map: dict[str, str],
    active_cte_cols: dict[str, tuple[str, ...]],
    local_ctes: list[tuple[tuple[str, ...], NormalizedQuery]],
) -> bool:
    """Alpha-rename CTEs wave-by-wave via Kahn's algorithm; return False on invalid body AST."""
    specs = _extract_cte_specs(with_node, dialect)
    if not specs:
        return True

    index_by_norm: dict[str, int] = {}
    for pos, spec in enumerate(specs):
        if spec.raw_norm not in index_by_norm:
            index_by_norm[spec.raw_norm] = pos

    deps: dict[int, set[int]] = {pos: set() for pos in range(len(specs))}
    for pos, spec in enumerate(specs):
        for tbl in spec.body_ast.find_all(exp.Table):
            if tbl.args.get("db") is not None or tbl.args.get("catalog") is not None:
                continue
            tbl_id = tbl.args.get("this")
            if isinstance(tbl_id, exp.Identifier) and tbl_id.name:
                ref_norm = normalize_grounding_identifier(
                    tbl_id.name, bool(tbl_id.quoted), dialect, is_table=True
                )
                dep_pos = index_by_norm.get(ref_norm)
                if dep_pos is not None and dep_pos != pos:
                    deps[pos].add(dep_pos)

    remaining = set(range(len(specs)))
    next_cte_idx = len(active_cte_map)

    while remaining:
        ready = [pos for pos in remaining if not (deps[pos] & remaining)]
        is_cycle_fallback = not ready
        batch = sorted(remaining) if is_cycle_fallback else ready

        wave_decomps: list[tuple[str, tuple[str, ...], _CteSpec, NormalizedQuery]] = []
        for pos in batch:
            spec = specs[pos]
            body_cte_map = dict(active_cte_map)
            body_cte_cols = dict(active_cte_cols)
            if is_cycle_fallback:
                for peer_pos in batch:
                    peer = specs[peer_pos]
                    tag = "__self__" if peer_pos == pos else "__cycle__"
                    body_cte_map[peer.raw_norm] = tag
                    body_cte_cols[tag] = peer.col_aliases
            else:
                body_cte_map[spec.raw_norm] = "__self__"
                body_cte_cols["__self__"] = spec.col_aliases

            body_nq = _decompose_query_ast(
                spec.body_ast,
                dialect=dialect,
                config=config,
                norm_schema=norm_schema,
                kmap=kmap,
                cte_name_map=body_cte_map,
                cte_col_aliases=body_cte_cols,
                outer_alias_bindings=None,
                outer_qual_bindings=None,
                outer_active_physical=frozenset(),
            )
            if body_nq is None:
                return False
            wave_decomps.append((body_nq.canonical_signature, spec.col_aliases, spec, body_nq))

        wave_decomps.sort(key=lambda item: (item[0], item[1]))
        for _, col_aliases, spec, body_nq in wave_decomps:
            placeholder = f"__cte_{next_cte_idx}__"
            next_cte_idx += 1
            active_cte_map[spec.raw_norm] = placeholder
            active_cte_cols[placeholder] = col_aliases
            local_ctes.append((col_aliases, body_nq))

        if is_cycle_fallback:
            remaining.clear()
        else:
            remaining.difference_update(ready)

    return True


def _set_op_kind(node: exp.SetOperation, *, ignore_distinct: bool) -> str:
    if isinstance(node, exp.Intersect):
        base = "intersect"
    elif isinstance(node, exp.Except):
        base = "except"
    else:
        base = "union"
    if ignore_distinct:
        return base
    is_distinct = node.args.get("distinct", True)
    return base if is_distinct is not False else f"{base}_all"


def _source_alias_spec(node: exp.Expr, dialect: str) -> tuple[str | None, str | None]:
    alias_node = node.args.get("alias")
    if isinstance(alias_node, exp.TableAlias) and isinstance(alias_node.this, exp.Identifier):
        name = alias_node.this.name
        quoted = bool(alias_node.this.quoted)
        return name, normalize_grounding_identifier(name, quoted, dialect, is_table=True)
    return None, None


def _table_part_info(part: exp.Expr | None, dialect: str) -> tuple[str | None, _CanonIdent | None]:
    if isinstance(part, exp.Identifier) and part.name:
        quoted = bool(part.quoted)
        norm = normalize_grounding_identifier(part.name, quoted, dialect, is_table=True)
        return norm, _canon_ident(part.name, quoted)
    return None, None


def _set_column_qualifier(col: exp.Column, canon_parts: _CanonTableParts) -> None:
    cat_canon, db_canon, tbl_canon = canon_parts
    col.set("table", _make_ident(tbl_canon))
    col.set("db", _make_ident(db_canon) if db_canon is not None else None)
    col.set("catalog", _make_ident(cat_canon) if cat_canon is not None else None)


def _lookup_qualified_target(
    col: exp.Column,
    dialect: str,
    local_aliases: Mapping[str, _AliasTarget],
    local_qual: Mapping[tuple[str | None, str | None, str], _AliasTarget],
    outer_aliases: Mapping[str, _AliasTarget],
    outer_qual: Mapping[tuple[str | None, str | None, str], _AliasTarget],
) -> tuple[_AliasTarget | None, bool]:
    """Resolve qualified `col` against local scope first, then outer correlated scopes."""
    parts = col.parts
    if len(parts) < 2:
        return None, False

    tbl_part = parts[-2]
    norm_tbl = normalize_grounding_identifier(
        tbl_part.name, bool(getattr(tbl_part, "quoted", False)), dialect, is_table=True
    )
    norm_db = (
        normalize_grounding_identifier(
            parts[-3].name,
            bool(getattr(parts[-3], "quoted", False)),
            dialect,
            is_table=True,
        )
        if len(parts) >= 3
        else None
    )
    norm_cat = (
        normalize_grounding_identifier(
            parts[-4].name,
            bool(getattr(parts[-4], "quoted", False)),
            dialect,
            is_table=True,
        )
        if len(parts) >= 4
        else None
    )

    for is_outer, aliases, qmap in (
        (False, local_aliases, local_qual),
        (True, outer_aliases, outer_qual),
    ):
        if norm_db is None and norm_cat is None:
            by_alias = aliases.get(norm_tbl)
            if by_alias is not None:
                return by_alias, is_outer
            direct_unqual = qmap.get((None, None, norm_tbl))
            if direct_unqual is not None:
                return direct_unqual, is_outer
            matches = [target for (_, _, t), target in qmap.items() if t == norm_tbl]
            if len(matches) == 1:
                return matches[0], is_outer
        else:
            exact = qmap.get((norm_cat, norm_db, norm_tbl))
            if exact is not None:
                return exact, is_outer
            if norm_cat is None:
                matches = [
                    target
                    for (c, d, t), target in qmap.items()
                    if d == norm_db and t == norm_tbl and c is not None
                ]
                if len(matches) == 1:
                    return matches[0], is_outer
    return None, False


def _column_in_schema_cols(col_norm: str, schema_cols: frozenset[str], dialect: str) -> bool:
    if col_norm in schema_cols:
        return True
    return dialect == "sqlite" and col_norm in _SQLITE_IMPLICIT_COLUMNS


def _is_exempt_from_value_mask(node: exp.Expr) -> bool:
    """Exempt non-value integer/type/limit/window literals from `ignore_values` masking."""
    parent = node.parent
    if (
        isinstance(node, exp.Literal)
        and not node.is_string
        and isinstance(node.this, str)
        and node.this.isdigit()
    ):
        if isinstance(parent, exp.Group) and node in parent.expressions:
            return True
        if isinstance(parent, exp.Ordered) and parent.this is node:
            return True

    exempt_classes = (
        exp.DataTypeParam,
        exp.DataType,
        exp.Limit,
        exp.Fetch,
        exp.Offset,
        exp.WindowSpec,
    )
    cur: exp.Expr | None = parent
    while cur is not None:
        if isinstance(cur, exempt_classes):
            return True
        if isinstance(cur, _SUBQUERY_BOUNDARY_CLASSES):
            return False
        cur = cur.parent
    return False


def _apply_value_mask_inplace(
    root: exp.Expr,
    *,
    ignore_distinct: bool,
    ignore_values: bool,
) -> None:
    """Handle `DISTINCT`, `COUNT(1)`, and `ignore_values` masking at the current level."""
    if ignore_distinct:
        root.set("distinct", None)
        for node in list(_walk_current_level(root)):
            if isinstance(node, exp.AggFunc) and isinstance(node.this, exp.Distinct):
                dist_exprs = [e.copy() for e in node.this.expressions]
                if dist_exprs:
                    node.set("this", dist_exprs[0])
                    if len(dist_exprs) > 1:
                        extra = [e.copy() for e in node.expressions]
                        node.set("expressions", dist_exprs[1:] + extra)

    if not ignore_values:
        return

    # Normalize COUNT(1) -> COUNT(*) before replacing numeric literals with __val__.
    for node in list(_walk_current_level(root)):
        if (
            isinstance(node, exp.Count)
            and isinstance(node.this, exp.Literal)
            and not node.this.is_string
            and node.this.this == "1"
            and not node.expressions
        ):
            node.set("this", exp.Star())

    # Mask negative numeric literals (Neg wrapping Literal) before bare literals.
    for node in list(_walk_current_level(root)):
        if (
            isinstance(node, exp.Neg)
            and isinstance(node.this, exp.Literal)
            and not node.this.is_string
            and not _is_exempt_from_value_mask(node)
        ):
            node.replace(exp.Var(this="__val__"))

    for node in list(_walk_current_level(root)):
        if (
            isinstance(node, (exp.Literal, exp.Boolean, exp.Null))
            and not _is_exempt_from_value_mask(node)
        ):
            node.replace(exp.Var(this="__val__"))


def _is_rhs_value_like(expr: exp.Expr) -> bool:
    if isinstance(expr, (exp.Literal, exp.Boolean, exp.Null, exp.Placeholder)):
        return True
    if isinstance(expr, exp.Neg) and isinstance(expr.this, exp.Literal):
        return True
    if isinstance(expr, exp.Var) and isinstance(expr.this, str):
        return expr.this == "__val__" or (
            expr.this.startswith("__subq_") and expr.this.endswith("__")
        )
    return False


def _flatten_assoc(node: exp.Binary, cls: type[exp.Binary]) -> list[exp.Expr]:
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


def _canonicalize_predicates_subtree(
    node: exp.Expr,
    dialect: str,
    *,
    skip_args: frozenset[str] = frozenset(),
) -> exp.Expr:
    """Canonicalize predicates bottom-up without descending into nested query boundaries."""
    if not skip_args and isinstance(node, _SUBQUERY_BOUNDARY_CLASSES):
        return node

    for key, val in list(node.args.items()):
        if key in skip_args:
            continue
        if isinstance(val, exp.Expr):
            if not isinstance(val, _SUBQUERY_BOUNDARY_CLASSES):
                new_val = _canonicalize_predicates_subtree(val, dialect)
                if new_val is not val:
                    node.set(key, new_val)
        elif isinstance(val, list):
            changed = False
            new_list: list[Any] = []
            for item in val:
                if isinstance(item, exp.Expr) and not isinstance(item, _SUBQUERY_BOUNDARY_CLASSES):
                    new_item = _canonicalize_predicates_subtree(item, dialect)
                    changed = changed or (new_item is not item)
                    new_list.append(new_item)
                else:
                    new_list.append(item)
            if changed:
                node.set(key, new_list)

    if isinstance(node, (exp.EQ, exp.NEQ)):
        if node.this is not None and node.expression is not None:
            left_val_like = _is_rhs_value_like(node.this)
            right_val_like = _is_rhs_value_like(node.expression)
            swap = False
            if left_val_like and not right_val_like:
                swap = True
            elif left_val_like == right_val_like:
                swap = _render_expr(node.this, dialect) > _render_expr(node.expression, dialect)
            if swap:
                left_copy = node.this.copy()
                right_copy = node.expression.copy()
                node.set("this", right_copy)
                node.set("expression", left_copy)
        return node

    node_type = type(node)
    if node_type in _INEQUALITY_FLIP and node.this is not None and node.expression is not None:
        if _is_rhs_value_like(node.this) and not _is_rhs_value_like(node.expression):
            flipped_cls = _INEQUALITY_FLIP[node_type]
            return flipped_cls(this=node.expression.copy(), expression=node.this.copy())
        return node

    if isinstance(node, exp.In):
        if node.args.get("query") is None and len(node.expressions) > 1:
            sorted_exprs = sorted(node.expressions, key=lambda e: _render_expr(e, dialect))
            node.set("expressions", [e.copy() for e in sorted_exprs])
        return node

    if isinstance(node, (exp.And, exp.Or)):
        cls = type(node)
        if _is_inside_same_assoc(node, cls):
            return node
        args = _flatten_assoc(node, cls)
        if len(args) > 1:
            args.sort(key=lambda expr: _render_expr(expr, dialect))
            rebuilt: exp.Expr = args[0].copy()
            for arg in args[1:]:
                rebuilt = cls(this=rebuilt, expression=arg.copy())
            return rebuilt
    return node


def _canonicalize_select_predicates(base_select: exp.Select, dialect: str) -> None:
    """Run predicate canonicalization on all current-level clauses, skipping FROM/JOIN targets."""
    _canonicalize_predicates_subtree(
        base_select, dialect, skip_args=frozenset({"from_", "joins", "with_"})
    )
    for join in base_select.args.get("joins", []):
        if isinstance(join, exp.Join):
            on_expr = join.args.get("on")
            if isinstance(on_expr, exp.Expr):
                new_on = _canonicalize_predicates_subtree(on_expr, dialect)
                if new_on is not on_expr:
                    join.set("on", new_on)


@dataclass
class _SubqueryRewriteContext:
    dialect: str
    config: MatchingConfig
    norm_schema: _NormalizedSchema | None
    kmap: _ForeignKeyMap
    active_cte_map: Mapping[str, str]
    active_cte_cols: Mapping[str, tuple[str, ...]]
    outer_aliases: Mapping[str, _AliasTarget]
    outer_qual: Mapping[tuple[str | None, str | None, str], _AliasTarget]
    outer_physical: frozenset[str]
    count: int = 0

    def decompose_placeholder(self, sub_ast: exp.Expr) -> str | None:
        sub_nq = _decompose_query_ast(
            sub_ast,
            dialect=self.dialect,
            config=self.config,
            norm_schema=self.norm_schema,
            kmap=self.kmap,
            cte_name_map=self.active_cte_map,
            cte_col_aliases=self.active_cte_cols,
            outer_alias_bindings=self.outer_aliases,
            outer_qual_bindings=self.outer_qual,
            outer_active_physical=self.outer_physical,
        )
        if sub_nq is None:
            return None
        self.count += 1
        return f"__subq_{sub_nq.canonical_signature}__"


def _replace_nested_subqueries_in_subtree(
    node: exp.Expr,
    ctx: _SubqueryRewriteContext,
    *,
    skip_args: frozenset[str] = frozenset(),
) -> bool:
    """Recursively replace nested queries in current-level clauses with `__subq_<sig>__`."""
    if isinstance(node, exp.Exists):
        target = node.this
        if isinstance(target, (exp.Select, exp.SetOperation, exp.Subquery)):
            placeholder = ctx.decompose_placeholder(target)
            if placeholder is None:
                return False
            node.set("this", exp.Var(this=placeholder))
        elif target is not None and not _replace_nested_subqueries_in_subtree(target, ctx):
            return False
        return True

    if isinstance(node, exp.In):
        if node.this is not None and not _replace_nested_subqueries_in_subtree(node.this, ctx):
            return False
        for item in node.expressions:
            if not _replace_nested_subqueries_in_subtree(item, ctx):
                return False
        q_node = node.args.get("query")
        if isinstance(q_node, (exp.Select, exp.SetOperation, exp.Subquery)):
            placeholder = ctx.decompose_placeholder(q_node)
            if placeholder is None:
                return False
            node.set("query", exp.Var(this=placeholder))
        return True

    for key, val in list(node.args.items()):
        if key in skip_args:
            continue
        if isinstance(val, exp.Expr):
            if isinstance(val, (exp.Select, exp.SetOperation, exp.Subquery)):
                placeholder = ctx.decompose_placeholder(val)
                if placeholder is None:
                    return False
                node.set(key, exp.Var(this=placeholder))
            elif not _replace_nested_subqueries_in_subtree(val, ctx):
                return False
        elif isinstance(val, list):
            new_list: list[Any] = []
            changed = False
            for item in val:
                if isinstance(item, (exp.Select, exp.SetOperation, exp.Subquery)):
                    placeholder = ctx.decompose_placeholder(item)
                    if placeholder is None:
                        return False
                    new_list.append(exp.Var(this=placeholder))
                    changed = True
                elif isinstance(item, exp.Expr):
                    if not _replace_nested_subqueries_in_subtree(item, ctx):
                        return False
                    new_list.append(item)
                else:
                    new_list.append(item)
            if changed:
                node.set(key, new_list)
    return True


def _split_top_level_and_or(node: exp.Expr | None) -> list[exp.Expr]:
    if node is None:
        return []
    cur = node
    while isinstance(cur, exp.Paren) and cur.this is not None:
        cur = cur.this
    if isinstance(cur, (exp.And, exp.Or)):
        out: list[exp.Expr] = []
        if cur.this is not None:
            out.extend(_split_top_level_and_or(cur.this))
        if cur.expression is not None:
            out.extend(_split_top_level_and_or(cur.expression))
        return out
    return [cur]


def _split_top_level_and_only(node: exp.Expr | None) -> list[exp.Expr]:
    if node is None:
        return []
    cur = node
    while isinstance(cur, exp.Paren) and isinstance(cur.this, (exp.And, exp.Paren)):
        cur = cur.this
    if isinstance(cur, exp.And):
        out: list[exp.Expr] = []
        if cur.this is not None:
            out.extend(_split_top_level_and_only(cur.this))
        if cur.expression is not None:
            out.extend(_split_top_level_and_only(cur.expression))
        return out
    return [cur]


def _collect_and_or_connectors(node: exp.Expr | None) -> frozenset[str]:
    if node is None:
        return frozenset()
    found: set[str] = set()
    for sub in _walk_current_level(node, skip_from_and_join_targets=False):
        if isinstance(sub, exp.And):
            found.add("and")
        elif isinstance(sub, exp.Or):
            found.add("or")
    return frozenset(found)


def _extract_where_no_op(atoms: Sequence[exp.Expr], dialect: str) -> tuple[str, ...]:
    targets: list[str] = []
    for atom in atoms:
        inner = atom
        while isinstance(inner, (exp.Not, exp.Paren)) and inner.this is not None:
            inner = inner.this
        if isinstance(inner, _WHERE_TARGET_OP_CLASSES) and inner.this is not None:
            targets.append(_render_expr(inner.this, dialect))
        elif isinstance(inner, exp.Exists):
            targets.append("exists")
        else:
            targets.append(_render_expr(inner, dialect))
    return tuple(targets)


def _extract_select_no_agg(expressions: Sequence[exp.Expr], dialect: str) -> tuple[str, ...]:
    items: list[str] = []
    for expr in expressions:
        cur = expr
        while isinstance(cur, exp.Paren) and cur.this is not None:
            cur = cur.this
        if isinstance(cur, _AGG_STRIP_CLASSES):
            arg_nodes: list[exp.Expr] = []
            if cur.this is not None:
                arg_nodes.append(cur.this)
            arg_nodes.extend(e for e in cur.expressions if isinstance(e, exp.Expr))
            if arg_nodes:
                items.append(", ".join(_render_expr(arg, dialect) for arg in arg_nodes))
            else:
                items.append(_render_expr(cur, dialect))
        else:
            items.append(_render_expr(cur, dialect))
    return tuple(items)


def _extract_condition_keywords_and_counts(
    base_select: exp.Select,
) -> tuple[set[str], int]:
    """Scan current-level JOIN ON, WHERE, and HAVING subtrees for keywords and hardness."""
    cond_roots: list[exp.Expr] = []
    for join in base_select.args.get("joins", []):
        if isinstance(join, exp.Join) and isinstance(join.args.get("on"), exp.Expr):
            cond_roots.append(join.args["on"])
    where_node = base_select.args.get("where")
    if isinstance(where_node, exp.Where) and isinstance(where_node.this, exp.Expr):
        cond_roots.append(where_node.this)
    having_node = base_select.args.get("having")
    if isinstance(having_node, exp.Having) and isinstance(having_node.this, exp.Expr):
        cond_roots.append(having_node.this)

    kw: set[str] = set()
    or_like_count = 0
    for root in cond_roots:
        for node in _walk_current_level(root, skip_from_and_join_targets=False):
            if isinstance(node, exp.Or):
                kw.add("or")
                or_like_count += 1
            if isinstance(node, exp.Not) or bool(node.args.get("negate")):
                kw.add("not")
            if isinstance(node, exp.In):
                kw.add("in")
            if isinstance(node, (exp.Like, exp.ILike)):
                kw.add("like")
                or_like_count += 1
    return kw, or_like_count


def _count_current_level_aggs(base_select: exp.Select) -> int:
    total = 0
    for clause_node in (
        list(base_select.expressions),
        base_select.args.get("where"),
        base_select.args.get("group"),
        base_select.args.get("order"),
        base_select.args.get("having"),
    ):
        roots = clause_node if isinstance(clause_node, list) else [clause_node]
        for root in roots:
            if not isinstance(root, exp.Expr):
                continue
            for node in _walk_current_level(root, skip_from_and_join_targets=False):
                if isinstance(node, exp.AggFunc):
                    total += 1
    return total


def _compute_hardness(
    *,
    where_preds: Sequence[str],
    group_by_full: Sequence[str],
    having_preds: Sequence[str],
    order_by: Sequence[tuple[str, str, str]],
    has_limit: bool,
    from_tables: Sequence[str],
    from_subqueries: Sequence[NormalizedQuery],
    or_like_count: int,
    direct_nested_query_count: int,
    iuen_branches: Sequence[tuple[str, NormalizedQuery]],
    ctes: Sequence[tuple[tuple[str, ...], NormalizedQuery]],
    agg_count: int,
    select_items: Sequence[str],
) -> str:
    count_comp1 = (
        int(bool(where_preds))
        + int(bool(group_by_full or having_preds))
        + int(bool(order_by))
        + int(has_limit)
        + max(0, len(from_tables) + len(from_subqueries) - 1)
        + or_like_count
    )
    count_comp2 = (
        len(from_subqueries) + direct_nested_query_count + len(iuen_branches) + len(ctes)
    )
    count_others = (
        int(agg_count > 1)
        + int(len(select_items) > 1)
        + int(len(where_preds) > 1)
        + int(len(group_by_full) > 1)
    )

    if count_comp1 <= 1 and count_others == 0 and count_comp2 == 0:
        return "easy"
    if (count_others <= 2 and count_comp1 <= 1 and count_comp2 == 0) or (
        count_comp1 <= 2 and count_others < 2 and count_comp2 == 0
    ):
        return "medium"
    if (
        (count_others > 2 and count_comp1 <= 2 and count_comp2 == 0)
        or (2 < count_comp1 <= 3 and count_others <= 2 and count_comp2 == 0)
        or (count_comp1 <= 1 and count_others == 0 and count_comp2 <= 1)
    ):
        return "hard"
    return "extra"


def _decompose_query_ast(
    ast: exp.Expr,
    *,
    dialect: str,
    config: MatchingConfig,
    norm_schema: _NormalizedSchema | None,
    kmap: _ForeignKeyMap,
    cte_name_map: Mapping[str, str] | None,
    cte_col_aliases: Mapping[str, tuple[str, ...]] | None,
    outer_alias_bindings: Mapping[str, _AliasTarget] | None,
    outer_qual_bindings: Mapping[tuple[str | None, str | None, str], _AliasTarget] | None,
    outer_active_physical: frozenset[str],
) -> NormalizedQuery | None:
    root = _unwrap_query_subquery(ast)
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        return None

    active_cte_map = dict(cte_name_map or {})
    active_cte_cols = dict(cte_col_aliases or {})
    local_ctes: list[tuple[tuple[str, ...], NormalizedQuery]] = []

    # Pass 1: Wave-by-wave CTE alpha-renaming on root.
    root_with = root.args.get("with_")
    if isinstance(root_with, exp.With):
        root.set("with_", None)
        if not _process_with_clause(
            root_with,
            dialect=dialect,
            config=config,
            norm_schema=norm_schema,
            kmap=kmap,
            active_cte_map=active_cte_map,
            active_cte_cols=active_cte_cols,
            local_ctes=local_ctes,
        ):
            return None

    # Pass 2: Left-associative IUEN flattening and root modifier hoisting.
    iuen_rev: list[tuple[str, NormalizedQuery]] = []
    collected_mods: dict[str, exp.Expr] = {}
    cur: exp.Expr = root
    while isinstance(cur, exp.SetOperation):
        cur_with = cur.args.get("with_")
        if isinstance(cur_with, exp.With):
            cur.set("with_", None)
            if not _process_with_clause(
                cur_with,
                dialect=dialect,
                config=config,
                norm_schema=norm_schema,
                kmap=kmap,
                active_cte_map=active_cte_map,
                active_cte_cols=active_cte_cols,
                local_ctes=local_ctes,
            ):
                return None
        for mod_key in _MODIFIER_KEYS:
            mod_val = cur.args.get(mod_key)
            if mod_val is not None and mod_key not in collected_mods:
                collected_mods[mod_key] = mod_val.copy()

        op_kind = _set_op_kind(cur, ignore_distinct=config.ignore_distinct)
        if cur.expression is None:
            return None
        rhs_nq = _decompose_query_ast(
            cur.expression,
            dialect=dialect,
            config=config,
            norm_schema=norm_schema,
            kmap=kmap,
            cte_name_map=active_cte_map,
            cte_col_aliases=active_cte_cols,
            outer_alias_bindings=outer_alias_bindings,
            outer_qual_bindings=outer_qual_bindings,
            outer_active_physical=outer_active_physical,
        )
        if rhs_nq is None:
            return None
        iuen_rev.append((op_kind, rhs_nq))
        if cur.this is None:
            return None
        cur = _unwrap_query_subquery(cur.this)

    if not isinstance(cur, exp.Select):
        return None
    base_select = cur

    left_with = base_select.args.get("with_")
    if isinstance(left_with, exp.With):
        base_select.set("with_", None)
        if not _process_with_clause(
            left_with,
            dialect=dialect,
            config=config,
            norm_schema=norm_schema,
            kmap=kmap,
            active_cte_map=active_cte_map,
            active_cte_cols=active_cte_cols,
            local_ctes=local_ctes,
        ):
            return None

    for mod_key, mod_val in collected_mods.items():
        base_select.set(mod_key, mod_val.copy())

    iuen_branches = tuple(reversed(iuen_rev))

    # Pass 3: FROM / JOIN resolution, eager derived tables, column qualification, kmap, folding.
    source_nodes: list[exp.Expr] = []
    from_node = base_select.args.get("from_")
    if isinstance(from_node, exp.From) and from_node.this is not None:
        source_nodes.append(from_node.this)
    for join in base_select.args.get("joins", []):
        if isinstance(join, exp.Join) and join.this is not None:
            source_nodes.append(join.this)

    local_aliases: dict[str, _AliasTarget] = {}
    local_qual: dict[tuple[str | None, str | None, str], _AliasTarget] = {}
    local_sources_list: list[_AliasTarget] = []
    local_active_physical: set[str] = set()
    from_tables: list[str] = []
    derived_entries: list[tuple[str, int, exp.Subquery, str | None, NormalizedQuery]] = []

    for src_idx, src in enumerate(source_nodes):
        if isinstance(src, exp.Table):
            _, alias_norm = _source_alias_spec(src, dialect)
            tbl_id = src.args.get("this")
            cat_node = src.args.get("catalog")
            db_node = src.args.get("db")
            if isinstance(tbl_id, exp.Identifier) and tbl_id.name:
                norm_tbl, canon_tbl = _table_part_info(tbl_id, dialect)
                assert norm_tbl is not None and canon_tbl is not None
                if cat_node is None and db_node is None and norm_tbl in active_cte_map:
                    placeholder = active_cte_map[norm_tbl]
                    src.set("alias", None)
                    src.set("this", exp.Identifier(this=placeholder, quoted=False))
                    from_tables.append(placeholder)
                    target = _AliasTarget(
                        kind="cte",
                        canon_parts=(None, None, (placeholder, False)),
                        norm_tbl=None,
                        schema_cols=None,
                        placeholder=placeholder,
                    )
                    key = alias_norm if alias_norm is not None else norm_tbl
                    local_aliases[key] = target
                    local_sources_list.append(target)
                else:
                    norm_cat, canon_cat = _table_part_info(cat_node, dialect)
                    norm_db, canon_db = _table_part_info(db_node, dialect)
                    src.set("this", _make_ident(canon_tbl))
                    if canon_db is not None:
                        src.set("db", _make_ident(canon_db))
                    if canon_cat is not None:
                        src.set("catalog", _make_ident(canon_cat))
                    src.set("alias", None)
                    from_tables.append(_render_expr(src, dialect))
                    local_active_physical.add(norm_tbl)
                    schema_cols = (
                        norm_schema.lookup(norm_cat, norm_db, norm_tbl)
                        if norm_schema is not None
                        else None
                    )
                    target = _AliasTarget(
                        kind="physical",
                        canon_parts=(canon_cat, canon_db, canon_tbl),
                        norm_tbl=norm_tbl,
                        schema_cols=schema_cols,
                        placeholder=None,
                    )
                    local_sources_list.append(target)
                    if alias_norm is not None:
                        local_aliases[alias_norm] = target
                    else:
                        local_qual[(norm_cat, norm_db, norm_tbl)] = target
            else:
                src.set("alias", None)
                for ident in src.find_all(exp.Identifier):
                    _fold_ident_inplace(ident)
                from_tables.append(_render_expr(src, dialect))
                target = _AliasTarget(
                    kind="opaque",
                    canon_parts=None,
                    norm_tbl=None,
                    schema_cols=None,
                    placeholder=None,
                )
                local_sources_list.append(target)
                if alias_norm is not None:
                    local_aliases[alias_norm] = target
        elif isinstance(src, exp.Subquery) and src.this is not None:
            _, alias_norm = _source_alias_spec(src, dialect)
            derived_nq = _decompose_query_ast(
                src.this,
                dialect=dialect,
                config=config,
                norm_schema=norm_schema,
                kmap=kmap,
                cte_name_map=active_cte_map,
                cte_col_aliases=active_cte_cols,
                outer_alias_bindings=outer_alias_bindings,
                outer_qual_bindings=outer_qual_bindings,
                outer_active_physical=outer_active_physical,
            )
            if derived_nq is None:
                return None
            derived_entries.append(
                (derived_nq.canonical_signature, src_idx, src, alias_norm, derived_nq)
            )
        else:
            _, alias_norm = _source_alias_spec(src, dialect)
            src.set("alias", None)
            for ident in src.find_all(exp.Identifier):
                _fold_ident_inplace(ident)
            from_tables.append(_render_expr(src, dialect))
            target = _AliasTarget(
                kind="opaque",
                canon_parts=None,
                norm_tbl=None,
                schema_cols=None,
                placeholder=None,
            )
            local_sources_list.append(target)
            if alias_norm is not None:
                local_aliases[alias_norm] = target

    derived_entries.sort(key=lambda item: (item[0], item[1]))
    from_subqueries_list: list[NormalizedQuery] = []
    for subq_idx, (_, _, subq_node, alias_norm, derived_nq) in enumerate(derived_entries):
        placeholder = f"__subq_{subq_idx}__"
        subq_node.set("alias", None)
        from_subqueries_list.append(derived_nq)
        target = _AliasTarget(
            kind="derived",
            canon_parts=(None, None, (placeholder, False)),
            norm_tbl=None,
            schema_cols=None,
            placeholder=placeholder,
        )
        local_sources_list.append(target)
        if alias_norm is not None:
            local_aliases[alias_norm] = target
    from_subqueries = tuple(from_subqueries_list)

    select_alias_set = _select_alias_names(base_select, dialect)
    sole_physical_target: _AliasTarget | None = (
        local_sources_list[0]
        if len(local_sources_list) == 1 and local_sources_list[0].kind == "physical"
        else None
    )
    outer_aliases = outer_alias_bindings or {}
    outer_qual = outer_qual_bindings or {}
    all_active_physical = outer_active_physical | frozenset(local_active_physical)

    for node in list(_walk_current_level(base_select)):
        if not isinstance(node, exp.Column):
            continue
        is_star_col = isinstance(node.this, exp.Star)
        resolved_target: _AliasTarget | None = None
        if node.table:
            resolved_target, _ = _lookup_qualified_target(
                node, dialect, local_aliases, local_qual, outer_aliases, outer_qual
            )
            if resolved_target is not None and resolved_target.canon_parts is not None:
                _set_column_qualifier(node, resolved_target.canon_parts)
        elif not is_star_col and isinstance(node.this, exp.Identifier) and node.this.name:
            col_norm = normalize_grounding_identifier(
                node.this.name, bool(node.this.quoted), dialect, is_table=False
            )
            if not (_in_clause_for_select(node, base_select) and col_norm in select_alias_set):
                if norm_schema is not None:
                    matching_targets = [
                        t
                        for t in local_sources_list
                        if t.kind == "physical"
                        and t.schema_cols is not None
                        and _column_in_schema_cols(col_norm, t.schema_cols, dialect)
                    ]
                    if len(matching_targets) == 1:
                        resolved_target = matching_targets[0]
                        assert resolved_target.canon_parts is not None
                        _set_column_qualifier(node, resolved_target.canon_parts)
                elif sole_physical_target is not None:
                    sole_parts = sole_physical_target.canon_parts
                    if sole_parts is not None:
                        resolved_target = sole_physical_target
                        _set_column_qualifier(node, sole_parts)

        # Apply foreign key kmap only when column resolves to an active physical table.
        if (
            kmap
            and not is_star_col
            and resolved_target is not None
            and resolved_target.kind == "physical"
            and resolved_target.norm_tbl is not None
            and resolved_target.norm_tbl in all_active_physical
            and isinstance(node.this, exp.Identifier)
            and node.this.name
        ):
            col_norm = normalize_grounding_identifier(
                node.this.name, bool(node.this.quoted), dialect, is_table=False
            )
            rep = kmap.get((resolved_target.norm_tbl, col_norm))
            if rep is not None:
                node.set("db", None)
                node.set("catalog", None)
                node.set("table", _make_ident(rep[0]))
                node.set("this", _make_ident(rep[1]))

    for node in _walk_current_level(base_select):
        if isinstance(node, exp.Identifier):
            _fold_ident_inplace(node)

    # Pass 4: SELECT alias stripping, DISTINCT normalization, and ignore_values masking.
    stripped_selects: list[exp.Expr] = []
    for item in base_select.expressions:
        if isinstance(item, exp.Alias) and item.this is not None:
            stripped_selects.append(item.this.copy())
        else:
            stripped_selects.append(item.copy())
    base_select.set("expressions", stripped_selects)

    _apply_value_mask_inplace(
        base_select,
        ignore_distinct=config.ignore_distinct,
        ignore_values=config.ignore_values,
    )

    # Pass 5: Recursive nested subquery replacement (excluding FROM/JOIN derived tables).
    subq_ctx = _SubqueryRewriteContext(
        dialect=dialect,
        config=config,
        norm_schema=norm_schema,
        kmap=kmap,
        active_cte_map=active_cte_map,
        active_cte_cols=active_cte_cols,
        outer_aliases={**outer_aliases, **local_aliases},
        outer_qual={**outer_qual, **local_qual},
        outer_physical=all_active_physical,
    )

    if not _replace_nested_subqueries_in_subtree(
        base_select, subq_ctx, skip_args=frozenset({"from_", "joins", "with_"})
    ):
        return None

    for join in base_select.args.get("joins", []):
        if isinstance(join, exp.Join):
            on_expr = join.args.get("on")
            if isinstance(on_expr, exp.Expr) and not _replace_nested_subqueries_in_subtree(
                on_expr, subq_ctx
            ):
                return None

    _canonicalize_select_predicates(base_select, dialect)

    # Pass 6: Extract 12 immutable clause representations, keywords, hardness, canonical signature.
    dist_node = base_select.args.get("distinct")
    select_distinct = (not config.ignore_distinct) and (dist_node is not None)
    select_distinct_on = ""
    if (
        select_distinct
        and isinstance(dist_node, exp.Distinct)
        and isinstance(dist_node.args.get("on"), exp.Expr)
    ):
        select_distinct_on = _render_expr(dist_node.args["on"], dialect)

    select_items = tuple(_render_expr(expr, dialect) for expr in base_select.expressions)
    select_no_agg = _extract_select_no_agg(base_select.expressions, dialect)

    where_node = base_select.args.get("where")
    where_root = where_node.this if isinstance(where_node, exp.Where) else None
    where_atoms = _split_top_level_and_or(where_root)
    where_preds = tuple(_render_expr(atom, dialect) for atom in where_atoms)
    where_no_op = _extract_where_no_op(where_atoms, dialect)
    where_and_or = _collect_and_or_connectors(where_root)

    group_node = base_select.args.get("group")
    group_exprs = (
        [e for e in group_node.expressions if isinstance(e, exp.Expr)]
        if isinstance(group_node, exp.Group)
        else []
    )
    group_by_full = tuple(_render_expr(g, dialect) for g in group_exprs)
    group_no_tbl_list: list[str] = []
    for g in group_exprs:
        if isinstance(g, exp.Column) and isinstance(g.this, exp.Identifier):
            stripped_col = g.copy()
            stripped_col.set("table", None)
            stripped_col.set("db", None)
            stripped_col.set("catalog", None)
            group_no_tbl_list.append(_render_expr(stripped_col, dialect))
        else:
            group_no_tbl_list.append(_render_expr(g, dialect))
    group_by_no_table = tuple(group_no_tbl_list)

    having_node = base_select.args.get("having")
    having_root = having_node.this if isinstance(having_node, exp.Having) else None
    having_atoms = _split_top_level_and_or(having_root)
    having_preds = tuple(_render_expr(atom, dialect) for atom in having_atoms)
    having_and_or = _collect_and_or_connectors(having_root)

    order_node = base_select.args.get("order")
    order_items: list[tuple[str, str, str]] = []
    if isinstance(order_node, exp.Order):
        for ordered in order_node.expressions:
            if isinstance(ordered, exp.Ordered) and ordered.this is not None:
                direction = "desc" if ordered.args.get("desc") else "asc"
                nf = ordered.args.get("nulls_first")
                nulls = "first" if nf is True else ("last" if nf is False else "default")
                order_items.append((_render_expr(ordered.this, dialect), direction, nulls))
    order_by = tuple(order_items)

    limit_node = base_select.args.get("limit")
    offset_node = base_select.args.get("offset")
    has_limit = (limit_node is not None) or (offset_node is not None)
    limit_value: str | None = None
    if config.include_limit_value and has_limit:
        lim_parts: list[str] = []
        if isinstance(limit_node, exp.Limit) and isinstance(limit_node.expression, exp.Expr):
            lim_parts.append(f"LIMIT {_render_expr(limit_node.expression, dialect)}")
        elif isinstance(limit_node, exp.Fetch):
            fetch_count = limit_node.args.get("count")
            if isinstance(fetch_count, exp.Expr):
                lim_parts.append(f"LIMIT {_render_expr(fetch_count, dialect)}")
        if isinstance(offset_node, exp.Offset) and isinstance(offset_node.expression, exp.Expr):
            lim_parts.append(f"OFFSET {_render_expr(offset_node.expression, dialect)}")
        limit_value = " ".join(lim_parts) if lim_parts else None

    join_on_items: list[str] = []
    if config.include_join_conditions:
        for join in base_select.args.get("joins", []):
            if not isinstance(join, exp.Join):
                continue
            on_expr = join.args.get("on")
            if isinstance(on_expr, exp.Expr):
                for atom in _split_top_level_and_only(on_expr):
                    join_on_items.append(_render_expr(atom, dialect))
            using_list = join.args.get("using")
            if isinstance(using_list, list):
                for u_item in using_list:
                    if isinstance(u_item, exp.Expr):
                        join_on_items.append(f"USING({_render_expr(u_item, dialect)})")
    join_on = tuple(join_on_items)

    ctes_tuple = tuple(local_ctes)
    from_tables_tuple = tuple(from_tables)

    cond_kw, or_like_count = _extract_condition_keywords_and_counts(base_select)
    kw: set[str] = set(cond_kw)
    if where_preds:
        kw.add("where")
    if group_by_full:
        kw.add("group")
    if having_preds:
        kw.add("having")
    if order_by:
        kw.add("order")
        if any(d == "asc" for _, d, _ in order_by):
            kw.add("asc")
        if any(d == "desc" for _, d, _ in order_by):
            kw.add("desc")
    if has_limit:
        kw.add("limit")
    for op, _ in iuen_branches:
        kw.add(op.split("_", 1)[0])
    if ctes_tuple:
        kw.add("with")
    keywords = frozenset(kw)

    agg_count = _count_current_level_aggs(base_select)
    hardness = _compute_hardness(
        where_preds=where_preds,
        group_by_full=group_by_full,
        having_preds=having_preds,
        order_by=order_by,
        has_limit=has_limit,
        from_tables=from_tables_tuple,
        from_subqueries=from_subqueries,
        or_like_count=or_like_count,
        direct_nested_query_count=subq_ctx.count,
        iuen_branches=iuen_branches,
        ctes=ctes_tuple,
        agg_count=agg_count,
        select_items=select_items,
    )

    sig = _compute_canonical_signature(
        is_empty=False,
        select_distinct=select_distinct,
        select_distinct_on=select_distinct_on,
        select_items=select_items,
        from_tables=from_tables_tuple,
        from_subqueries=from_subqueries,
        join_on=join_on,
        where_preds=where_preds,
        where_and_or=where_and_or,
        group_by_full=group_by_full,
        having_preds=having_preds,
        having_and_or=having_and_or,
        order_by=order_by,
        has_limit=has_limit,
        limit_value=limit_value,
        iuen_branches=iuen_branches,
        ctes=ctes_tuple,
        keywords=keywords,
    )

    return NormalizedQuery(
        is_empty=False,
        select_distinct=select_distinct,
        select_distinct_on=select_distinct_on,
        select_items=select_items,
        select_no_agg=select_no_agg,
        from_tables=from_tables_tuple,
        from_subqueries=from_subqueries,
        join_on=join_on,
        where_preds=where_preds,
        where_no_op=where_no_op,
        where_and_or=where_and_or,
        group_by_no_table=group_by_no_table,
        group_by_full=group_by_full,
        having_preds=having_preds,
        having_and_or=having_and_or,
        order_by=order_by,
        has_limit=has_limit,
        limit_value=limit_value,
        iuen_branches=iuen_branches,
        ctes=ctes_tuple,
        keywords=keywords,
        hardness=hardness,
        canonical_signature=sig,
    )


def decompose_query_ast(
    ast: exp.Expr,
    *,
    dialect: str,
    config: MatchingConfig,
    norm_schema: _NormalizedSchema | None,
    kmap: _ForeignKeyMap,
) -> NormalizedQuery | None:
    """Decompose a validated SQL query AST into an immutable `NormalizedQuery`."""
    try:
        return _decompose_query_ast(
            ast,
            dialect=dialect,
            config=config,
            norm_schema=norm_schema,
            kmap=kmap,
            cte_name_map=None,
            cte_col_aliases=None,
            outer_alias_bindings=None,
            outer_qual_bindings=None,
            outer_active_physical=frozenset(),
        )
    except (SqlglotError, RecursionError):
        return None


_EMPTY_SIG = _compute_canonical_signature(
    is_empty=True,
    select_distinct=False,
    select_distinct_on="",
    select_items=(),
    from_tables=(),
    from_subqueries=(),
    join_on=(),
    where_preds=(),
    where_and_or=frozenset(),
    group_by_full=(),
    having_preds=(),
    having_and_or=frozenset(),
    order_by=(),
    has_limit=False,
    limit_value=None,
    iuen_branches=(),
    ctes=(),
    keywords=frozenset(),
)

_EMPTY_QUERY = NormalizedQuery(
    is_empty=True,
    select_distinct=False,
    select_distinct_on="",
    select_items=(),
    select_no_agg=(),
    from_tables=(),
    from_subqueries=(),
    join_on=(),
    where_preds=(),
    where_no_op=(),
    where_and_or=frozenset(),
    group_by_no_table=(),
    group_by_full=(),
    having_preds=(),
    having_and_or=frozenset(),
    order_by=(),
    has_limit=False,
    limit_value=None,
    iuen_branches=(),
    ctes=(),
    keywords=frozenset(),
    hardness="easy",
    canonical_signature=_EMPTY_SIG,
)


def empty_query_decomposition() -> NormalizedQuery:
    """Return an immutable empty decomposition used for unparseable predictions."""
    return _EMPTY_QUERY


def _get_scores(cnt: int, pred_total: int, label_total: int) -> ComponentScore:
    if pred_total != label_total or cnt != pred_total:
        return ComponentScore(0, 0, 0, label_total, pred_total)
    return ComponentScore(1, 1, 1, label_total, pred_total)


def _multiset_str_overlap(pred_items: Sequence[str], ref_items: Sequence[str]) -> int:
    if not pred_items or not ref_items:
        return 0
    return sum((Counter(pred_items) & Counter(ref_items)).values())


def _multiset_match_count(
    preds: Sequence[_T],
    refs: Sequence[_T],
    eq_fn: Callable[[_T, _T], bool],
) -> int:
    if not preds or not refs:
        return 0
    used = [False] * len(refs)
    matched = 0
    for pred_item in preds:
        for idx, ref_item in enumerate(refs):
            if not used[idx] and eq_fn(pred_item, ref_item):
                used[idx] = True
                matched += 1
                break
    return matched


def score_pair(
    pred: NormalizedQuery,
    ref: NormalizedQuery,
    config: MatchingConfig,
    *,
    prediction_parsed: bool = True,
    schema_applied: bool = False,
    foreign_keys_applied: bool = False,
) -> PairMatchResult:
    """Score one prediction decomposition against one reference decomposition."""
    components: dict[str, ComponentScore] = {}

    # 1. select & 2. select(no AGG)
    sel_cnt = _multiset_str_overlap(pred.select_items, ref.select_items)
    if not config.ignore_distinct and (
        pred.select_distinct != ref.select_distinct
        or pred.select_distinct_on != ref.select_distinct_on
    ):
        sel_cnt = 0
    components["select"] = _get_scores(sel_cnt, len(pred.select_items), len(ref.select_items))
    components["select(no AGG)"] = _get_scores(
        _multiset_str_overlap(pred.select_no_agg, ref.select_no_agg),
        len(pred.select_no_agg),
        len(ref.select_no_agg),
    )

    # 3. where & 4. where(no OP)
    components["where"] = _get_scores(
        _multiset_str_overlap(pred.where_preds, ref.where_preds),
        len(pred.where_preds),
        len(ref.where_preds),
    )
    components["where(no OP)"] = _get_scores(
        _multiset_str_overlap(pred.where_no_op, ref.where_no_op),
        len(pred.where_no_op),
        len(ref.where_no_op),
    )

    # 5. group(no Having) & 6. group
    components["group(no Having)"] = _get_scores(
        _multiset_str_overlap(pred.group_by_no_table, ref.group_by_no_table),
        len(pred.group_by_no_table),
        len(ref.group_by_no_table),
    )
    pred_grp_total = 1 if (pred.group_by_full or pred.having_preds) else 0
    ref_grp_total = 1 if (ref.group_by_full or ref.having_preds) else 0
    grp_cnt = (
        1
        if (
            pred_grp_total == ref_grp_total == 1
            and Counter(pred.group_by_full) == Counter(ref.group_by_full)
            and Counter(pred.having_preds) == Counter(ref.having_preds)
            and pred.having_and_or == ref.having_and_or
        )
        else 0
    )
    components["group"] = _get_scores(grp_cnt, pred_grp_total, ref_grp_total)

    # 7. order
    pred_ord_total = 1 if (pred.order_by or pred.has_limit) else 0
    ref_ord_total = 1 if (ref.order_by or ref.has_limit) else 0
    ord_cnt = (
        1
        if (
            pred_ord_total == ref_ord_total == 1
            and pred.order_by == ref.order_by
            and pred.has_limit == ref.has_limit
            and (not config.include_limit_value or pred.limit_value == ref.limit_value)
        )
        else 0
    )
    components["order"] = _get_scores(ord_cnt, pred_ord_total, ref_ord_total)

    # 8. and/or (bug-fixed argument order and non-inflated empty-set handling)
    if not pred.where_and_or and not ref.where_and_or:
        components["and/or"] = ComponentScore(1, 1, 1, 0, 0)
    else:
        ao_label_total = len(ref.where_and_or)
        ao_pred_total = len(pred.where_and_or)
        ao_cnt = ao_label_total if pred.where_and_or == ref.where_and_or else 0
        components["and/or"] = _get_scores(ao_cnt, ao_pred_total, ao_label_total)

    # 9. IUEN
    iuen_cnt = _multiset_match_count(
        pred.iuen_branches,
        ref.iuen_branches,
        lambda a, b: a[0] == b[0] and eval_exact_match(a[1], b[1], config) == 1,
    )
    components["IUEN"] = _get_scores(
        iuen_cnt, len(pred.iuen_branches), len(ref.iuen_branches)
    )

    # 10. keywords
    components["keywords"] = _get_scores(
        len(pred.keywords & ref.keywords),
        len(pred.keywords),
        len(ref.keywords),
    )

    # 11. cte
    cte_cnt = _multiset_match_count(
        pred.ctes,
        ref.ctes,
        lambda a, b: a[0] == b[0] and eval_exact_match(a[1], b[1], config) == 1,
    )
    components["cte"] = _get_scores(cte_cnt, len(pred.ctes), len(ref.ctes))

    # 12. join_on
    if not config.include_join_conditions:
        components["join_on"] = ComponentScore(1, 1, 1, 0, 0)
    else:
        components["join_on"] = _get_scores(
            _multiset_str_overlap(pred.join_on, ref.join_on),
            len(pred.join_on),
            len(ref.join_on),
        )

    exact = 0
    if (
        pred.is_empty == ref.is_empty
        and all(cs.f1 == 1 for cs in components.values())
        and Counter(pred.from_tables) == Counter(ref.from_tables)
        and len(pred.from_subqueries) == len(ref.from_subqueries)
        and _multiset_match_count(
            pred.from_subqueries,
            ref.from_subqueries,
            lambda a, b: eval_exact_match(a, b, config) == 1,
        )
        == len(ref.from_subqueries)
    ):
        exact = 1

    return PairMatchResult(
        exact_match=exact,
        hardness=ref.hardness,
        components=components,
        prediction_parsed=prediction_parsed,
        schema_applied=schema_applied,
        foreign_keys_applied=foreign_keys_applied,
    )


def eval_exact_match(
    pred: NormalizedQuery,
    ref: NormalizedQuery,
    config: MatchingConfig,
) -> int:
    """Return 1 if `pred` and `ref` match on all components and `FROM` tables/subqueries."""
    return score_pair(pred, ref, config).exact_match


def select_best_reference(
    pred: NormalizedQuery,
    refs: Sequence[NormalizedQuery],
    config: MatchingConfig,
    *,
    prediction_parsed: bool,
    schema_applied: bool,
    foreign_keys_applied: bool,
) -> PairMatchResult:
    """Select the deterministically highest-scoring reference decomposition for one example."""
    if not refs:
        raise ValueError("`refs` must contain at least one NormalizedQuery.")

    best_key: tuple[int, int, int, int] | None = None
    best_result: PairMatchResult | None = None
    for ref_idx, ref_nq in enumerate(refs):
        res = score_pair(
            pred,
            ref_nq,
            config,
            prediction_parsed=prediction_parsed,
            schema_applied=schema_applied,
            foreign_keys_applied=foreign_keys_applied,
        )
        matched_active = sum(
            1
            for cs in res.components.values()
            if (cs.label_total > 0 or cs.pred_total > 0) and cs.f1 == 1
        )
        all_f1_sum = sum(cs.f1 for cs in res.components.values())
        rank_key = (res.exact_match, matched_active, all_f1_sum, -ref_idx)
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_result = res

    assert best_result is not None
    return best_result


def finalize_component_bucket(
    acc_sum: int,
    acc_count: int,
    rec_sum: int,
    rec_count: int,
) -> dict[str, float | int | None]:
    """Finalize one component's precision/recall/F1 without Spider's zero-match F1=1 bug."""
    if acc_count == 0 and rec_count == 0:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "pred_count": 0,
            "label_count": 0,
        }
    precision = acc_sum / acc_count if acc_count > 0 else 0.0
    recall = rec_sum / rec_count if rec_count > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_count": acc_count,
        "label_count": rec_count,
    }


@dataclass(frozen=True)
class EvaluatedRow:
    """Outcome of parsing, decomposing, and scoring one example row."""

    parsed_refs: int
    total_refs: int
    prediction_parsed: bool
    match_result: PairMatchResult | None


def validate_single_row_inputs(
    prediction: Any,
    reference: Any,
    schema: SchemaDict | None,
    foreign_keys: ForeignKeysLike | None,
    *,
    dialect: str,
    extractor_enabled: bool,
) -> tuple[list[str], _NormalizedSchema | None, _ForeignKeyMap]:
    """Validate and coerce single-row evaluator inputs before mutating state."""
    if not extractor_enabled and not isinstance(prediction, str):
        raise TypeError(f"`prediction` must be a str, got {type(prediction).__name__}.")
    ref_group = coerce_reference_group(reference)
    norm_schema = normalize_schema(schema, dialect=dialect) if schema is not None else None
    kmap = normalize_foreign_keys(foreign_keys, dialect=dialect)
    return ref_group, norm_schema, kmap


def _validate_non_mapping_sequence(name: str, values: Any) -> Sequence[Any]:
    if isinstance(values, (str, bytes, bytearray, Mapping)) or not isinstance(values, Sequence):
        raise TypeError(
            f"`{name}` must be a non-str/non-mapping Sequence, got {type(values).__name__}."
        )
    return values


def validate_batch_inputs(
    predictions: Any,
    references: Any,
    schemas: Any,
    foreign_keys: Any,
    *,
    dialect: str,
    extractor_enabled: bool,
) -> list[tuple[Any, list[str], _NormalizedSchema | None, _ForeignKeyMap]]:
    """Eagerly validate batch sequences, lengths, schemas, and FKs before mutating state."""
    preds = _validate_non_mapping_sequence("predictions", predictions)
    refs = _validate_non_mapping_sequence("references", references)
    if len(preds) != len(refs):
        raise ValueError(
            f"Length mismatch: got {len(preds)} predictions and {len(refs)} references."
        )

    if schemas is None:
        norm_schemas: list[_NormalizedSchema | None] = [None] * len(preds)
    else:
        schema_seq = _validate_non_mapping_sequence("schemas", schemas)
        if len(preds) != len(schema_seq):
            raise ValueError(
                f"Length mismatch: got {len(preds)} predictions and {len(schema_seq)} schemas."
            )
        norm_schemas = [
            normalize_schema(schema, dialect=dialect) if schema is not None else None
            for schema in schema_seq
        ]

    if foreign_keys is None:
        kmaps: list[_ForeignKeyMap] = [{}] * len(preds)
    else:
        fk_seq = _validate_non_mapping_sequence("foreign_keys", foreign_keys)
        if len(preds) != len(fk_seq):
            raise ValueError(
                f"Length mismatch: got {len(preds)} predictions and {len(fk_seq)} foreign_keys."
            )
        kmaps = [normalize_foreign_keys(fk, dialect=dialect) for fk in fk_seq]

    if not extractor_enabled:
        for idx, pred in enumerate(preds):
            if not isinstance(pred, str):
                raise TypeError(
                    f"`predictions[{idx}]` must be str, got {type(pred).__name__}."
                )

    coerced_refs = [coerce_reference_group(ref, index=idx) for idx, ref in enumerate(refs)]
    return list(zip(preds, coerced_refs, norm_schemas, kmaps, strict=True))


def evaluate_matching_row(
    prediction: Any,
    ref_group: Sequence[str],
    norm_schema: _NormalizedSchema | None,
    kmap: _ForeignKeyMap,
    *,
    config: MatchingConfig,
    compiled_extractor: CompiledExtractor,
    max_query_chars: int | None,
    max_input_chars: int | None,
) -> EvaluatedRow:
    """Parse prediction & references, decompose ASTs, and score the best matching reference."""
    ref_decomps: list[NormalizedQuery] = []
    for ref_sql in ref_group:
        parsed_ref = parse_query_ast(
            ref_sql,
            dialect=config.dialect,
            max_query_chars=max_query_chars,
            max_input_chars=max_input_chars,
            compiled_extractor=None,
            apply_extract=False,
        )
        if parsed_ref.ok and parsed_ref.ast is not None:
            ref_nq = decompose_query_ast(
                parsed_ref.ast,
                dialect=config.dialect,
                config=config,
                norm_schema=norm_schema,
                kmap=kmap,
            )
            if ref_nq is not None:
                ref_decomps.append(ref_nq)

    parsed_pred = parse_query_ast(
        prediction,
        dialect=config.dialect,
        max_query_chars=max_query_chars,
        max_input_chars=max_input_chars,
        compiled_extractor=compiled_extractor,
        apply_extract=True,
    )
    pred_nq: NormalizedQuery | None = None
    if parsed_pred.ok and parsed_pred.ast is not None:
        pred_nq = decompose_query_ast(
            parsed_pred.ast,
            dialect=config.dialect,
            config=config,
            norm_schema=norm_schema,
            kmap=kmap,
        )
    pred_ok = pred_nq is not None

    if not ref_decomps or (not pred_ok and config.skip_unparseable_predictions):
        return EvaluatedRow(
            parsed_refs=len(ref_decomps),
            total_refs=len(ref_group),
            prediction_parsed=pred_ok,
            match_result=None,
        )

    effective_pred = pred_nq if pred_nq is not None else empty_query_decomposition()
    res = select_best_reference(
        effective_pred,
        ref_decomps,
        config,
        prediction_parsed=pred_ok,
        schema_applied=norm_schema is not None,
        foreign_keys_applied=bool(kmap),
    )
    return EvaluatedRow(
        parsed_refs=len(ref_decomps),
        total_refs=len(ref_group),
        prediction_parsed=pred_ok,
        match_result=res,
    )


class _BaseMatchingEvaluator(BaseEvaluator):
    """Shared streaming lifecycle, validation, and counters for matching evaluators."""

    def __init__(
        self,
        dialect: str = "postgres",
        *,
        ignore_values: bool = True,
        ignore_distinct: bool = True,
        include_limit_value: bool = False,
        include_join_conditions: bool = False,
        skip_unparseable_predictions: bool = False,
        extractor: ExtractorLike = None,
        max_query_chars: int | None = 65_536,
        max_input_chars: int | None = 131_072,
        include_per_example: bool = False,
    ) -> None:
        super().__init__(
            dialect=dialect,
            extractor=extractor,
            max_query_chars=max_query_chars,
            max_input_chars=max_input_chars,
            include_per_example=include_per_example,
        )
        self.ignore_values = bool(ignore_values)
        self.ignore_distinct = bool(ignore_distinct)
        self.include_limit_value = bool(include_limit_value)
        self.include_join_conditions = bool(include_join_conditions)
        self.skip_unparseable_predictions = bool(skip_unparseable_predictions)
        self._config = MatchingConfig(
            dialect=self.dialect,
            ignore_values=self.ignore_values,
            ignore_distinct=self.ignore_distinct,
            include_limit_value=self.include_limit_value,
            include_join_conditions=self.include_join_conditions,
            skip_unparseable_predictions=self.skip_unparseable_predictions,
        )
        self._scored_count = 0
        self._parsed_count = 0
        self._unparseable_count = 0
        self._total_refs_count = 0
        self._parsed_refs_count = 0
        self._schema_applied_count = 0
        self._foreign_keys_applied_count = 0
        self._hardness_counts: dict[str, int] = {}
        self._reset_common()

    def _reset_common(self) -> None:
        self._num_examples = 0
        self._scored_count = 0
        self._parsed_count = 0
        self._unparseable_count = 0
        self._total_refs_count = 0
        self._parsed_refs_count = 0
        self._schema_applied_count = 0
        self._foreign_keys_applied_count = 0
        self._hardness_counts = {level: 0 for level in ALL_HARDNESS_LEVELS}

    @abstractmethod
    def _record_match_result(self, res: PairMatchResult | None) -> None:
        """Update metric-specific accumulators after scoring one example row."""

    def _record_row(
        self,
        prediction: Any,
        ref_group: Sequence[str],
        norm_schema: _NormalizedSchema | None,
        kmap: _ForeignKeyMap,
    ) -> None:
        evaluated = evaluate_matching_row(
            prediction,
            ref_group,
            norm_schema,
            kmap,
            config=self._config,
            compiled_extractor=self._compiled_extractor,
            max_query_chars=self.max_query_chars,
            max_input_chars=self.max_input_chars,
        )
        self._num_examples += 1
        self._total_refs_count += evaluated.total_refs
        self._parsed_refs_count += evaluated.parsed_refs
        if evaluated.prediction_parsed:
            self._parsed_count += 1
        else:
            self._unparseable_count += 1

        res = evaluated.match_result
        if res is not None:
            self._scored_count += 1
            if res.schema_applied:
                self._schema_applied_count += 1
            if res.foreign_keys_applied:
                self._foreign_keys_applied_count += 1
            self._hardness_counts[res.hardness] += 1
            self._hardness_counts["all"] += 1
        self._record_match_result(res)

    def add(
        self,
        prediction: str,
        reference: str | Sequence[str],
        schema: SchemaDict | None = None,
        foreign_keys: ForeignKeysLike | None = None,
    ) -> None:
        """Add one prediction, reference(s), optional schema, and optional foreign keys."""
        ref_group, norm_schema, kmap = validate_single_row_inputs(
            prediction,
            reference,
            schema,
            foreign_keys,
            dialect=self.dialect,
            extractor_enabled=self._compiled_extractor is not None,
        )
        self._record_row(prediction, ref_group, norm_schema, kmap)

    def add_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str] | Sequence[Sequence[str]],
        schemas: Sequence[SchemaDict | None] | None = None,
        foreign_keys: Sequence[ForeignKeysLike | None] | None = None,
    ) -> None:
        """Eagerly validate and accumulate a batch of predictions, references, and metadata."""
        items = validate_batch_inputs(
            predictions,
            references,
            schemas,
            foreign_keys,
            dialect=self.dialect,
            extractor_enabled=self._compiled_extractor is not None,
        )
        for pred, ref_group, norm_schema, kmap in items:
            self._record_row(pred, ref_group, norm_schema, kmap)

    def compute(self) -> dict[str, Any]:
        """Compute accumulated matching metrics and reset state."""
        return super().compute()

    def _common_metadata_payload(self) -> dict[str, Any]:
        return {
            "scored_count": self._scored_count,
            "parsed_count": self._parsed_count,
            "unparseable_count": self._unparseable_count,
            "parse_success_rate": self._parsed_count / self._num_examples,
            "reference_parse_success_rate": (
                self._parsed_refs_count / self._total_refs_count
                if self._total_refs_count > 0
                else 0.0
            ),
            "schema_applied_count": self._schema_applied_count,
            "foreign_keys_applied_count": self._foreign_keys_applied_count,
            "num_examples": self._num_examples,
            "dialect": self.dialect,
            "ignore_values": self.ignore_values,
            "ignore_distinct": self.ignore_distinct,
            "include_limit_value": self.include_limit_value,
            "include_join_conditions": self.include_join_conditions,
            "skip_unparseable_predictions": self.skip_unparseable_predictions,
        }


