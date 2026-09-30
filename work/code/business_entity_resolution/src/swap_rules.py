"""Co-located sibling rule for France: one S1-vocabulary noun swapped at the same address.

France S1 names are '<prefix> <noun> <legal form>', and the test data holds many co-located
businesses that share the prefix and the address but carry another noun ('Gospel Sportive SARL'
vs 'Gospel Club SARL', same street and number). For 43-66% of such records another S1 carries
exactly the record's name at that number (they are separate businesses), against 0-15% for the
generator's filler words ('X SARL Développement', 'X Services', '& Fils', ...), which are true
noisy copies (US analog: 'center'/'services' swaps, 92.5% true). In US/India the same pattern is
rare and ~1% true, so the US/India-trained models never learned it and give France mean p 0.26.

vocab_swap_cap() caps p for these pairs; the filler words and stopwords are excluded.
"""
import polars as pl
from rapidfuzz import fuzz, process

from io_utils import load_records

LEGAL = (r"\b(sas|sasu|sarl|eurl|sa|sci|snc|ei|cie|inc|llc|ltd|co|corp|corporation|incorporated|lp|llp|pvt|"
         r"private|limited|company|pc|pllc|s a r l|s a s|s a)\b")
TOK = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
       .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(LEGAL, " ").str.split(" ")
       .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique().list.sort())
NUM = pl.col("business_address").fill_null("").str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1")
STRIP = (r"\b(rue|r|avenue|av|ave|boulevard|bd|blvd|place|pl|impasse|imp|allee|all|chemin|ch|quai|cours|crs|square|sq|"
         r"residence|res|route|rte|chaussee|passage|terrasse|de|du|des|la|le|les|l|d|no|n|appt|bis|ter|st|street|road|rd|"
         r"drive|dr|lane|ln|court|ct|circle|cir|way|unit|apt|suite|ste|fl|pmb|null|na)\b")
ADDR = (pl.col("business_address").fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(STRIP, " ").str.replace_all(r"\s+", " ").str.strip_chars())
FILLERS = {"US": ["center", "services", "service", "partners"], "India": ["center", "services", "service", "partners"],
           "France": ["services", "developpement", "groupe", "associes", "fils", "france"]}
SIBLING_ALSO = ["france", "developpement", "groupe"]  # fillers that are also sibling-generator qualifiers
STOP = ["and", "the", "of", "et", "de", "du", "des", "la", "le", "les", "au", "aux", "en", "sur"]  # '&' <-> 'and'/'et'
VOCAB_MIN = 300  # a vocabulary noun appears in >= this many S1 names of the country
VOCAB_MIN_EXT = 30
TYPO_MAX = 70    # fuzz.ratio(added, missing) below this = a different word, not a typo
ASIM_MIN = 85    # token-set similarity of the stripped addresses: same street and town
P_CAP = 0.003


def vocab_swap_pairs(pred: pl.DataFrame, split: str, country: str, extended: bool = False) -> pl.DataFrame:
    """(s1_id, cand_id) pairs: same number and street, names differ by one vocabulary noun.

    extended=True widens the rule (after the leaderboard confirmed it: France +0.0122 F0.5):
    nouns seen in >= VOCAB_MIN_EXT S1 names (S1 names are clean, so these are real words), the
    street match is dropped (the same number with a typo'd street is still the same site); both
    words must have >= 4 letters (short tokens are initials that US abbreviations collide with).
    """
    q = load_records(split, "source1", country).select(
        pl.col("entity_id").alias("s1_id"), TOK.alias("qt"), NUM.alias("qn"), ADDR.alias("qa"))
    counts = q.select(pl.col("qt").explode().alias("w")).group_by("w").len()
    vocab_base = counts.filter(pl.col("len") >= VOCAB_MIN)["w"].implode()
    vocab = counts.filter(pl.col("len") >= (VOCAB_MIN_EXT if extended else VOCAB_MIN))["w"].implode()
    r = pl.concat([load_records(split, s, country) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"), NUM.alias("tn"), ADDR.alias("ta"))
    same = (pred.select("s1_id", "cand_id").join(q, on="s1_id").join(r, on="cand_id")
            .filter(pl.col("qn").is_not_null() & (pl.col("qn") == pl.col("tn"))))
    d = (same.with_columns(add=pl.col("tt").list.set_difference(pl.col("qt")),
                           miss=pl.col("qt").list.set_difference(pl.col("tt")))
         .filter((pl.col("add").list.len() == 1) & (pl.col("miss").list.len() == 1))
         .with_columns(a=pl.col("add").list.first(), m=pl.col("miss").list.first()))
    fill = FILLERS.get(country, [])
    d = d.filter(pl.col("a").is_in(vocab) & pl.col("m").is_in(vocab) & ~pl.col("a").is_in(fill) & ~pl.col("m").is_in(fill)
                 & ~pl.col("a").is_in(STOP) & ~pl.col("m").is_in(STOP))
    sim = process.cpdist(d["a"].to_list(), d["m"].to_list(), scorer=fuzz.ratio, workers=-1)
    asim = process.cpdist(d["qa"].to_list(), d["ta"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    d = d.with_columns(sim=pl.Series(sim, dtype=pl.Float64), asim=pl.Series(asim, dtype=pl.Float64))
    keep = (pl.col("sim") < TYPO_MAX) & (pl.col("asim") >= ASIM_MIN)
    if extended:  # real nouns only: 2-3 letter tokens are initials that US abbreviations collide with ('care' -> 'ce')
        wide = (pl.col("sim") < TYPO_MAX) & (pl.col("a").str.len_chars() >= 4) & (pl.col("m").str.len_chars() >= 4)
        base = keep & pl.col("a").is_in(vocab_base) & pl.col("m").is_in(vocab_base)
        keep = base | wide
    return d.filter(keep).select("s1_id", "cand_id")


def filler_swap_pairs(pred: pl.DataFrame, split: str, country: str) -> pl.DataFrame:
    """(s1_id, cand_id) pairs that are the generator's filler variant of the S1 name (true noisy copies).

    One filler word added (with or without dropping one S1 word), same house number and street, and no
    other S1 carries exactly the record's name at that number. The US analog of this slice is 99.5% true.
    Fillers that are also sibling-generator words ('X Développement' at a nearby number is a sibling
    business) only count when they replace a word.
    """
    q = load_records(split, "source1", country).select(
        pl.col("entity_id").alias("s1_id"), TOK.alias("qt"), NUM.alias("qn"), ADDR.alias("qa"))
    r = pl.concat([load_records(split, s, country) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"), NUM.alias("tn"), ADDR.alias("ta"))
    d = (pred.select("s1_id", "cand_id").join(q, on="s1_id").join(r, on="cand_id")
         .filter(pl.col("qn").is_not_null() & (pl.col("qn") == pl.col("tn")))
         .with_columns(add=pl.col("tt").list.set_difference(pl.col("qt")),
                       miss=pl.col("qt").list.set_difference(pl.col("tt")))
         .filter((pl.col("add").list.len() == 1) & (pl.col("miss").list.len() <= 1))
         .with_columns(a=pl.col("add").list.first(), swap=pl.col("miss").list.len())
         .filter(pl.col("a").is_in(FILLERS.get(country, [])) & ((pl.col("swap") == 1) | ~pl.col("a").is_in(SIBLING_ALSO))))
    twins = q.select(pl.col("qt").list.join(" ").alias("k"), pl.col("qn").alias("tn")).unique()
    d = d.with_columns(k=pl.col("tt").list.join(" ")).join(twins, on=["k", "tn"], how="anti")
    asim = process.cpdist(d["qa"].to_list(), d["ta"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    return d.with_columns(asim=pl.Series(asim, dtype=pl.Float64)).filter(pl.col("asim") >= ASIM_MIN).select("s1_id", "cand_id")


def vocab_swap_cap(pred: pl.DataFrame, split: str, country: str, extended: bool = False) -> pl.DataFrame:
    """Return pred (one country) with p capped at P_CAP on vocabulary-swap pairs."""
    hit = vocab_swap_pairs(pred, split, country, extended).with_columns(_hit=pl.lit(True))
    return (pred.join(hit, on=["s1_id", "cand_id"], how="left")
            .with_columns(pl.when(pl.col("_hit")).then(pl.min_horizontal(pl.col("p"), pl.lit(P_CAP)))
                          .otherwise(pl.col("p")).cast(pred["p"].dtype).alias("p"))
            .drop("_hit"))
