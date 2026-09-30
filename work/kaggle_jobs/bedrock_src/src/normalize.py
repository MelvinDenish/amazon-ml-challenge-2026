"""Vectorised (polars) normalisation of business names and addresses.

Everything here is rule-based and derived from inspecting the training data
(no external lookups). The noise patterns handled are the ones observed in
train S2/S3: alias markers ("X f/k/a Y"), junk wrappers ("(ID: 123)", "#70318",
"***", "[...]"), domains/handles, legal-suffix variants, accented Latin letters,
street-type abbreviations, ordinal words, zero-padded / '##'-prefixed numbers
and <NULL> placeholders.
"""

import polars as pl

# Alias markers: the text AFTER the marker is the name that matches S1.
ALIAS_RE = (
    r"^(.*?)[\s,]*\b(?:formerly known as|formerly|f/k/a|fka|t/a|d/b/a|dba|a/k/a|aka|"
    r"trading as|doing business as)\b[\s:,-]*(.+)$"
)

# Tokens with no identifying value for blocking/matching of names.
NAME_STOP = {
    "private", "pvt", "limited", "ltd", "llc", "l", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llp", "plc", "pc", "pllc", "lp", "the", "and",
    "of", "m", "s", "ms", "sarl", "sas", "sasu", "eurl", "snc", "sa", "ei", "sci",
    "public", "pty", "cie", "et", "de", "du", "des", "la", "le", "les", "www", "com",
    "net", "org", "in", "fr", "service", "services", "center", "centre", "id",
}

# Canonical forms for common address words (US, India and France).
ADDR_CANON = {
    "street": "st", "saint": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "circle": "cir", "place": "pl", "lane": "ln", "court": "ct",
    "boulevard": "blvd", "bd": "blvd", "highway": "hwy", "parkway": "pkwy", "trail": "trl",
    "terrace": "ter", "square": "sq", "mount": "mt", "north": "n", "south": "s",
    "east": "e", "west": "w", "apartment": "apt", "suite": "ste", "sainte": "ste",
    "r": "rue", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte",
    "number": "no", "house": "h", "floor": "fl", "near": "nr", "opposite": "opp",
    "nagar": "ngr", "colony": "col", "sector": "sec",
}
ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20",
}
ADDR_STOP = {
    "st", "rd", "ave", "dr", "cir", "pl", "ln", "ct", "blvd", "hwy", "pkwy", "trl", "ter",
    "sq", "n", "s", "e", "w", "apt", "ste", "unit", "pmb", "po", "box", "no", "rue", "de",
    "du", "des", "la", "le", "les", "h", "fl", "nr", "opp", "near", "null", "c", "o",
    "ward", "at", "post", "dist", "district", "tq", "taluk", "the", "and", "of", "floor",
    "ground", "first", "shop", "plot", "flat", "building", "bldg", "room",
}


def _base_clean(col: pl.Expr) -> pl.Expr:
    """Lower-case and strip accents from Latin letters only.

    Indic combining marks are kept, because they are part of the letter.
    """
    return (
        col.str.normalize("NFKD")
        .str.replace_all(r"(\p{Latin})\p{Mn}+", "$1")
        .str.to_lowercase()
    )


def _tokens_expr(col: pl.Expr) -> pl.Expr:
    """Replace every non letter/number/mark run with one space and trim."""
    return col.str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ").str.strip_chars()


def normalize_names(df: pl.DataFrame, col: str = "business_name") -> pl.DataFrame:
    """Add name columns used by blocking and features.

    n_full   cleaned full name (all tokens, legal words kept)
    n_core   cleaned name after alias marker if present, junk removed
    n_alias  the random prefix before an alias marker ("" if none)
    n_toks   list of informative tokens of n_core (stop words removed)
    n_nospace  informative tokens of n_core concatenated (domain/handle comparable)
    is_native  name contains Indic script
    has_alias / is_domain flags
    """
    base = _base_clean(pl.col(col))
    junk = (
        base.str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")
        .str.replace_all(r"#\s*\d+", " ")
        .str.replace_all(r"\bwww\.", " ")
        .str.replace_all(r"\.(com|co\.in|in|fr|net|org|biz)\b", " ")
        .str.replace_all(r"&", " and ")
    )
    out = df.with_columns(
        _n_junk=junk,
        is_native=pl.col(col).str.contains(r"[\p{Devanagari}\p{Bengali}\p{Kannada}\p{Tamil}"
                                           r"\p{Telugu}\p{Gujarati}\p{Gurmukhi}\p{Malayalam}\p{Oriya}]"),
        is_domain=base.str.contains(r"\.(com|co\.in|in|fr|net|org|biz)\b|^\W*@"),
    )
    alias = pl.col("_n_junk").str.extract_groups(ALIAS_RE)
    out = out.with_columns(
        has_alias=pl.col("_n_junk").str.contains(ALIAS_RE),
        _a=alias.struct.field("1"),
        _c=alias.struct.field("2"),
    )
    out = out.with_columns(
        n_full=_tokens_expr(pl.col("_n_junk")),
        n_core=_tokens_expr(pl.when(pl.col("has_alias")).then(pl.col("_c")).otherwise(pl.col("_n_junk"))),
        n_alias=_tokens_expr(pl.when(pl.col("has_alias")).then(pl.col("_a")).otherwise(pl.lit(""))),
    )
    stop = list(NAME_STOP)
    out = out.with_columns(
        n_toks=pl.col("n_core").str.split(" ").list.eval(
            pl.element().filter((pl.element().str.len_chars() > 0) & ~pl.element().is_in(stop))
        )
    )
    out = out.with_columns(n_nospace=pl.col("n_toks").list.join(""))
    return out.drop("_n_junk", "_a", "_c")


def normalize_addresses(df: pl.DataFrame, col: str = "business_address") -> pl.DataFrame:
    """Add address columns used by blocking and features.

    a_norm    cleaned address with canonical street words, ordinals and
              zero-padding removed ("##801" -> "801", "0520" -> "520",
              "eleventh" -> "11", "11th" -> "11")
    a_toks    list of tokens of a_norm
    a_num     first house-number-like token ("" if none)
    a_street  first alphabetic, non-stop token after a_num ("" if none)
    a_empty   address empty or a null placeholder
    """
    base = (
        _base_clean(pl.col(col))
        .str.replace_all(r"<\s*null\s*>|\bnull\b|\bnone\b|\bn/a\b", " ")
        .str.replace_all(r"#+", " ")
    )
    toks = _tokens_expr(base).str.split(" ")
    canon_keys = list(ADDR_CANON) + list(ORDINALS)
    canon_vals = list(ADDR_CANON.values()) + list(ORDINALS.values())
    toks = toks.list.eval(
        pl.element()
        .str.replace(r"^0+(\d)", "$1")
        .str.replace(r"^(\d+)(st|nd|rd|th)$", "$1")
        .replace(canon_keys, canon_vals)
    ).list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
    out = df.with_columns(a_toks=toks)
    out = out.with_columns(
        a_norm=pl.col("a_toks").list.join(" "),
        a_empty=pl.col("a_toks").list.len() == 0,
    )
    # House number, then the first street word after it, skipping street-type and
    # French article words ("28 rue general chanzy" -> 28 / general; "520 11 st" -> 520 / 11).
    num_street = pl.col("a_norm").str.extract_groups(
        r"\b(\d+[a-z]?)\s+(?:(?:rue|ave|blvd|ch|all|imp|rte|de|du|des|la|le|les|n|s|e|w|no)\s+)*"
        r"([a-z]{3,}|\d+)"
    )
    return out.with_columns(
        a_num=num_street.struct.field("1").fill_null(""),
        a_street=num_street.struct.field("2").fill_null(""),
    )


def normalize_records(df: pl.DataFrame) -> pl.DataFrame:
    """Apply name and address normalisation to a raw record frame."""
    return normalize_addresses(normalize_names(df))


NATIVE_RE = (r"[\p{Devanagari}\p{Bengali}\p{Kannada}\p{Tamil}\p{Telugu}\p{Gujarati}"
             r"\p{Gurmukhi}\p{Malayalam}\p{Oriya}]")


def transliterate(df: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Rewrite native-script tokens with the train-learned map and refresh derived columns.

    ``was_native`` keeps the original script flag (a feature); ``is_native``
    becomes True only if native tokens remain after mapping.
    """
    natives, latins = mapping["native"].to_list(), mapping["latin"].to_list()

    def mapped(c: str) -> pl.Expr:
        return (
            pl.col(c).str.split(" ")
            .list.eval(pl.element().replace(natives, latins))
            .list.join(" ").alias(c)
        )

    stop = list(NAME_STOP)
    out = df.with_columns(pl.col("is_native").alias("was_native"))
    out = out.with_columns(mapped("n_core"), mapped("n_full"), mapped("a_norm"))
    return out.with_columns(
        is_native=pl.col("n_core").str.contains(NATIVE_RE),
        n_toks=pl.col("n_core").str.split(" ").list.eval(
            pl.element().filter((pl.element().str.len_chars() > 0) & ~pl.element().is_in(stop))),
        a_toks=pl.col("a_norm").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)),
    ).with_columns(n_nospace=pl.col("n_toks").list.join(""))
