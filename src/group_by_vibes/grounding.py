"""Static query-scope schema grounding using `sqlglot.optimizer.scope`."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, ScopeType, build_scope

from .dialects import (
    normalize_dialect,
    normalize_grounding_identifier,
    split_qualified_table_parts,
)

_SQLITE_IMPLICIT_COLUMNS = frozenset({"rowid", "_rowid_", "oid"})


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of checking a parsed AST against a per-query schema dict."""

    ok: bool
    failure_reason: str | None
    missing_tables: tuple[str, ...]
    missing_columns: tuple[str, ...]
    skipped: bool = False


class _TableKey(NamedTuple):
    catalog: str | None
    db: str | None
    table: str


class _NormalizedSchema:
    """Per-query normalized schema catalog (not cached across calls)."""

    def __init__(self, dialect: str, exact_tables: dict[_TableKey, frozenset[str]]) -> None:
        self.dialect = dialect
        self.exact_tables = exact_tables
        self.by_table_name: dict[str, list[tuple[_TableKey, frozenset[str]]]] = defaultdict(list)
        for key, cols in exact_tables.items():
            self.by_table_name[key.table].append((key, cols))

    def lookup(
        self,
        catalog: str | None,
        db: str | None,
        table: str,
    ) -> frozenset[str] | None:
        """Resolve `(catalog, db, table)` without falling back across mismatched qualifiers."""
        if catalog is not None and db is not None:
            return self.exact_tables.get(_TableKey(catalog, db, table))
        if db is not None:
            direct = self.exact_tables.get(_TableKey(None, db, table))
            if direct is not None:
                return direct
            matches = [cols for key, cols in self.by_table_name.get(table, ()) if key.db == db]
            return matches[0] if len(matches) == 1 else None
        direct_unqual = self.exact_tables.get(_TableKey(None, None, table))
        if direct_unqual is not None:
            return direct_unqual
        candidates = self.by_table_name.get(table, ())
        return candidates[0][1] if len(candidates) == 1 else None


class _SourceBinding(NamedTuple):
    opaque: bool
    columns: frozenset[str]
    is_physical: bool
    is_missing_table: bool


def _normalize_schema_key_part(
    part: tuple[str, bool] | None,
    dialect: str,
) -> str | None:
    if part is None:
        return None
    name, quoted = part
    if dialect == "sqlite":
        return name.lower()
    if quoted:
        return name
    strategy = Dialect.get_or_raise(dialect).NORMALIZATION_STRATEGY
    if strategy == NormalizationStrategy.CASE_INSENSITIVE:
        return name.lower()
    return name


def _normalize_schema_column(col_name: str, dialect: str) -> str:
    if dialect == "sqlite":
        return col_name.lower()
    strategy = Dialect.get_or_raise(dialect).NORMALIZATION_STRATEGY
    if strategy in (
        NormalizationStrategy.CASE_INSENSITIVE,
        NormalizationStrategy.CASE_INSENSITIVE_EXCEPT_TABLES,
    ):
        return col_name.lower()
    return col_name


def normalize_schema(
    schema: Mapping[str, Iterable[str]],
    dialect: str = "postgres",
) -> _NormalizedSchema:
    """Validate and normalize a per-query `{table: columns}` mapping."""
    if not isinstance(schema, Mapping):
        raise TypeError(f"`schema` must be a Mapping or None, got {type(schema).__name__}.")
    norm_dialect = normalize_dialect(dialect)
    exact_tables: dict[_TableKey, frozenset[str]] = {}

    for raw_table, raw_cols in schema.items():
        if not isinstance(raw_table, str):
            raise TypeError(f"Schema table keys must be str, got {type(raw_table).__name__}.")
        if isinstance(raw_cols, (str, bytes)) or not isinstance(raw_cols, Iterable):
            raise TypeError(
                f"Columns for table {raw_table!r} must be a non-str Iterable of str, "
                f"got {type(raw_cols).__name__}."
            )
        cat_part, db_part, tbl_part = split_qualified_table_parts(raw_table, dialect=norm_dialect)
        table_key = _TableKey(
            catalog=_normalize_schema_key_part(cat_part, norm_dialect),
            db=_normalize_schema_key_part(db_part, norm_dialect),
            table=_normalize_schema_key_part(tbl_part, norm_dialect) or tbl_part[0],
        )
        col_set: set[str] = set()
        for col in raw_cols:
            if not isinstance(col, str) or not col:
                raise ValueError(
                    f"Schema column names for {raw_table!r} must be non-empty strings, got {col!r}."
                )
            col_set.add(_normalize_schema_column(col, norm_dialect))
        merged = exact_tables.get(table_key, frozenset()) | frozenset(col_set)
        exact_tables[table_key] = merged

    return _NormalizedSchema(norm_dialect, exact_tables)


def _ident_parts(node: exp.Expr | None) -> tuple[str | None, bool]:
    if isinstance(node, exp.Identifier):
        return node.name, bool(node.quoted)
    return None, False


def _table_display_name(tbl: exp.Table) -> str:
    parts = [p for p in (tbl.catalog, tbl.db, tbl.name) if p]
    return ".".join(parts) if parts else tbl.name


def _scope_child_binding(source_scope: Scope, dialect: str) -> _SourceBinding:
    parent = source_scope.expression.parent
    alias_node = parent.args.get("alias") if isinstance(parent, exp.Expr) else None
    if isinstance(alias_node, exp.TableAlias) and alias_node.columns:
        cols = frozenset(
            normalize_grounding_identifier(c.name, bool(c.quoted), dialect)
            for c in alias_node.columns
            if isinstance(c, exp.Identifier) and c.name
        )
        return _SourceBinding(opaque=False, columns=cols, is_physical=False, is_missing_table=False)

    if source_scope.scope_type == ScopeType.CTE and isinstance(source_scope.expression, exp.Select):
        proj_names: set[str] = set()
        for expr in source_scope.expression.expressions:
            if isinstance(expr, exp.Star):
                return _SourceBinding(
                    opaque=True, columns=frozenset(), is_physical=False, is_missing_table=False
                )
            if isinstance(expr, exp.Alias) and isinstance(expr.args.get("alias"), exp.Identifier):
                alias_id = expr.args["alias"]
                proj_names.add(
                    normalize_grounding_identifier(alias_id.name, bool(alias_id.quoted), dialect)
                )
            elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Identifier):
                col_id = expr.this
                proj_names.add(
                    normalize_grounding_identifier(col_id.name, bool(col_id.quoted), dialect)
                )
            else:
                return _SourceBinding(
                    opaque=True, columns=frozenset(), is_physical=False, is_missing_table=False
                )
        return _SourceBinding(
            opaque=False,
            columns=frozenset(proj_names),
            is_physical=False,
            is_missing_table=False,
        )

    return _SourceBinding(
        opaque=True, columns=frozenset(), is_physical=False, is_missing_table=False
    )


def _source_alias_key(source_node: exp.Table | Scope, raw_key: str, dialect: str) -> str:
    if isinstance(source_node, exp.Table):
        alias_node = source_node.args.get("alias")
        if isinstance(alias_node, exp.TableAlias) and isinstance(alias_node.this, exp.Identifier):
            return normalize_grounding_identifier(
                alias_node.this.name, bool(alias_node.this.quoted), dialect, is_table=True
            )
        tbl_id = source_node.args.get("this")
        if isinstance(tbl_id, exp.Identifier):
            return normalize_grounding_identifier(
                tbl_id.name, bool(tbl_id.quoted), dialect, is_table=True
            )
        return normalize_grounding_identifier(raw_key, False, dialect, is_table=True)

    parent = source_node.expression.parent
    alias_node = parent.args.get("alias") if isinstance(parent, exp.Expr) else None
    if (
        isinstance(alias_node, exp.TableAlias)
        and isinstance(alias_node.this, exp.Identifier)
        and alias_node.this.name == raw_key
    ):
        return normalize_grounding_identifier(
            alias_node.this.name, bool(alias_node.this.quoted), dialect, is_table=True
        )
    return normalize_grounding_identifier(raw_key, False, dialect, is_table=True)


def _in_clause_for_select(col: exp.Column, select_expr: exp.Expr) -> bool:
    if not isinstance(select_expr, exp.Select):
        return False
    cur: exp.Expr | None = col.parent
    in_allowed_clause = False
    while cur is not None:
        if cur is select_expr:
            return in_allowed_clause
        if isinstance(cur, (exp.Group, exp.Having, exp.Order)):
            in_allowed_clause = True
        elif isinstance(cur, (exp.Select, exp.Subquery, exp.CTE)):
            return False
        cur = cur.parent
    return False


def _select_alias_names(expr: exp.Expr, dialect: str) -> frozenset[str]:
    if not isinstance(expr, exp.Select):
        return frozenset()
    out: set[str] = set()
    for item in expr.expressions:
        if isinstance(item, exp.Alias) and isinstance(item.args.get("alias"), exp.Identifier):
            alias_id = item.args["alias"]
            out.add(normalize_grounding_identifier(alias_id.name, bool(alias_id.quoted), dialect))
    return frozenset(out)


def _column_in_binding(
    col_norm: str,
    binding: _SourceBinding,
    dialect: str,
) -> bool:
    if binding.is_missing_table or binding.opaque:
        return True
    if col_norm in binding.columns:
        return True
    return (
        binding.is_physical
        and dialect == "sqlite"
        and col_norm in _SQLITE_IMPLICIT_COLUMNS
    )


def _find_qualified_binding(
    col: exp.Column,
    bindings_by_scope: dict[int, dict[str, _SourceBinding]],
    table_qual_by_scope: dict[int, dict[tuple[str | None, str | None, str], _SourceBinding]],
    scope: Scope,
    dialect: str,
) -> _SourceBinding | None:
    tbl_part = col.parts[-2]
    norm_tbl = normalize_grounding_identifier(
        tbl_part.name, bool(getattr(tbl_part, "quoted", False)), dialect, is_table=True
    )
    norm_db = (
        normalize_grounding_identifier(
            col.parts[-3].name,
            bool(getattr(col.parts[-3], "quoted", False)),
            dialect,
            is_table=True,
        )
        if len(col.parts) >= 3
        else None
    )
    norm_cat = (
        normalize_grounding_identifier(
            col.parts[-4].name,
            bool(getattr(col.parts[-4], "quoted", False)),
            dialect,
            is_table=True,
        )
        if len(col.parts) >= 4
        else None
    )

    cur: Scope | None = scope
    while cur is not None:
        if norm_db is None and norm_cat is None:
            found = bindings_by_scope[id(cur)].get(norm_tbl)
            if found is not None:
                return found
        else:
            qmap = table_qual_by_scope[id(cur)]
            exact = qmap.get((norm_cat, norm_db, norm_tbl))
            if exact is not None:
                return exact
            if norm_cat is None:
                matches = [
                    b
                    for (c, d, t), b in qmap.items()
                    if d == norm_db and t == norm_tbl and c is not None
                ]
                if len(matches) == 1:
                    return matches[0]
        cur = cur.parent
    return None


def check_schema_grounding(
    ast: exp.Expr,
    normalized_schema: _NormalizedSchema,
    dialect: str = "postgres",
) -> GroundingResult:
    """Validate table and column existence on query AST scopes."""
    norm_dialect = normalize_dialect(dialect)
    try:
        root_scope = build_scope(ast)
    except (SqlglotError, RecursionError):
        return GroundingResult(
            ok=False,
            failure_reason="syntax_error",
            missing_tables=(),
            missing_columns=(),
            skipped=False,
        )
    if root_scope is None:
        return GroundingResult(
            ok=True,
            failure_reason=None,
            missing_tables=(),
            missing_columns=(),
            skipped=True,
        )

    missing_tables: set[str] = set()
    missing_columns: set[str] = set()

    try:
        scopes = list(root_scope.traverse())
        bindings_by_scope: dict[int, dict[str, _SourceBinding]] = {}
        table_qual_by_scope: dict[
            int, dict[tuple[str | None, str | None, str], _SourceBinding]
        ] = {}
        select_aliases_by_scope: dict[int, frozenset[str]] = {}

        for sc in scopes:
            local_bindings: dict[str, _SourceBinding] = {}
            local_qual: dict[tuple[str | None, str | None, str], _SourceBinding] = {}
            for raw_key, src in sc.sources.items():
                alias_key = _source_alias_key(src, raw_key, norm_dialect)
                if isinstance(src, exp.Table):
                    cat_n, cat_q = _ident_parts(src.args.get("catalog"))
                    db_n, db_q = _ident_parts(src.args.get("db"))
                    tbl_n, tbl_q = _ident_parts(src.args.get("this"))
                    norm_cat = (
                        normalize_grounding_identifier(cat_n, cat_q, norm_dialect, is_table=True)
                        if cat_n
                        else None
                    )
                    norm_db = (
                        normalize_grounding_identifier(db_n, db_q, norm_dialect, is_table=True)
                        if db_n
                        else None
                    )
                    norm_tbl = normalize_grounding_identifier(
                        tbl_n or src.name, tbl_q, norm_dialect, is_table=True
                    )
                    found_cols = normalized_schema.lookup(norm_cat, norm_db, norm_tbl)
                    if found_cols is None:
                        missing_tables.add(_table_display_name(src))
                        binding = _SourceBinding(
                            opaque=False,
                            columns=frozenset(),
                            is_physical=True,
                            is_missing_table=True,
                        )
                    else:
                        binding = _SourceBinding(
                            opaque=False,
                            columns=found_cols,
                            is_physical=True,
                            is_missing_table=False,
                        )
                    local_bindings[alias_key] = binding
                    if not src.alias:
                        local_qual[(norm_cat, norm_db, norm_tbl)] = binding
                elif isinstance(src, Scope):
                    binding = _scope_child_binding(src, norm_dialect)
                    local_bindings[alias_key] = binding

            bindings_by_scope[id(sc)] = local_bindings
            table_qual_by_scope[id(sc)] = local_qual
            select_aliases_by_scope[id(sc)] = _select_alias_names(sc.expression, norm_dialect)

        checked_col_ids: set[int] = set()
        for sc in scopes:
            local_bindings = bindings_by_scope[id(sc)]
            for star in sc.stars:
                if isinstance(star, exp.Column) and star.table:
                    tbl_part = star.parts[-2] if len(star.parts) >= 2 else star.args.get("table")
                    q_name = star.table
                    q_quoted = bool(getattr(tbl_part, "quoted", False))
                    q_norm = normalize_grounding_identifier(
                        q_name, q_quoted, norm_dialect, is_table=True
                    )
                    if q_norm not in local_bindings:
                        missing_tables.add(q_name)

            for col in sc.columns:
                if id(col) in checked_col_ids or isinstance(col.this, exp.Star):
                    continue
                checked_col_ids.add(id(col))
                col_id = col.this
                col_quoted = isinstance(col_id, exp.Identifier) and bool(col_id.quoted)
                col_norm = normalize_grounding_identifier(col.name, col_quoted, norm_dialect)

                if col.table and len(col.parts) >= 2:
                    found_binding = _find_qualified_binding(
                        col, bindings_by_scope, table_qual_by_scope, sc, norm_dialect
                    )
                    if found_binding is None:
                        missing_tables.add(".".join(p.name for p in col.parts[:-1]))
                    elif not _column_in_binding(col_norm, found_binding, norm_dialect):
                        missing_columns.add(f"{col.table}.{col.name}")
                    continue

                if (
                    _in_clause_for_select(col, sc.expression)
                    and col_norm in select_aliases_by_scope[id(sc)]
                ):
                    continue

                resolved = False
                cur: Scope | None = sc
                while cur is not None:
                    cur_bindings = bindings_by_scope[id(cur)].values()
                    if any(_column_in_binding(col_norm, b, norm_dialect) for b in cur_bindings):
                        resolved = True
                        break
                    if (
                        _in_clause_for_select(col, cur.expression)
                        and col_norm in select_aliases_by_scope[id(cur)]
                    ):
                        resolved = True
                        break
                    cur = cur.parent

                if not resolved:
                    missing_columns.add(col.name)
    except (SqlglotError, RecursionError):
        return GroundingResult(
            ok=False,
            failure_reason="syntax_error",
            missing_tables=(),
            missing_columns=(),
            skipped=False,
        )

    mt = tuple(sorted(missing_tables))
    mc = tuple(sorted(missing_columns))
    if mt:
        return GroundingResult(
            ok=False,
            failure_reason="unknown_table",
            missing_tables=mt,
            missing_columns=mc,
            skipped=False,
        )
    if mc:
        return GroundingResult(
            ok=False,
            failure_reason="unknown_column",
            missing_tables=(),
            missing_columns=mc,
            skipped=False,
        )
    return GroundingResult(
        ok=True,
        failure_reason=None,
        missing_tables=(),
        missing_columns=(),
        skipped=False,
    )
