"""Word-swap rules transferred from the labelled US structure to France.

Measured on held-out US (real labels) and on France test (no labels), for pairs whose S2/S3 name
differs from the S1 name by exactly one word and whose house number is the same:

* NOISE swap (the generator appends a filler word and drops the last noun: 'Womens Health Partners
  LLC' -> 'Womens Health LLC Center'). US fillers {center, services, service, partners}: 160/1k S1,
  92.5% true when no other S1 carries the record's exact name. France has the same 160-170/1k volume
  with fillers {services, developpement, groupe, associes, fils, france} (uniform 6-8k each, 4-250x
  their S1-vocabulary frequency), but the US-trained models give them mean p 0.58 (US 0.92).
* NOUN swap (one S1-vocabulary noun replaced by another: 'Pole Patrimoine SARL' -> 'Pole Club SARL'
  at the same address). France has ~425/1k S1 of these; each noun appears in proportion to its S1
  frequency, and 1,043/1k have an S1 twin carrying exactly the record's name at that number
  (co-located sibling businesses). In US they are rare and 0.3% true. The models give France mean
  p 0.12-0.2 and select ~46/1k S1 of them.

    python france_swap_rules.py eval <evaloof.parquet>                     # US/India held-out check
    python france_swap_rules.py test <pred.parquet> <out.parquet> [noun|noise|both]
"""
import sys

import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs, load_records  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from score import per_entity_f05  # noqa: E402

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
ASIM_MIN = 85        # token-set similarity of the stripped addresses: same street and town
NOISE = {"US": ["center", "services", "service", "partners"], "India": ["center", "services", "service", "partners"],
         "France": ["services", "developpement", "groupe", "associes", "fils", "france"]}
SIBLING_ALSO = ["france", "developpement", "groupe"]  # also sibling-generator words: only trust them as a swap
VOCAB_MIN = 300      # a 'vocabulary noun' appears in >= this many S1 names of the country
TYPO_MAX = 70        # fuzz.ratio(added, missing) below this = a different word, not a typo
P_NOISE = 0.92       # US truth rate of no-twin same-number filler swaps
P_NOUN = 0.003       # US truth rate of same-number vocabulary swaps
STOP = ["and", "the", "of", "et", "de", "du", "des", "la", "le", "les", "au", "aux", "en", "sur"]  # '&' <-> 'and'/'et'


def swap_pairs(split: str, c: str, pred: pl.DataFrame) -> pl.DataFrame:
    """(s1_id, cand_id, kind) for the pairs one of the rules applies to."""
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), TOK.alias("qt"), NUM.alias("qn"),
                                                 ADDR.alias("qa"))
    vocab = (q.select(pl.col("qt").explode().alias("w")).group_by("w").len()
             .filter(pl.col("len") >= VOCAB_MIN)["w"].implode())
    r = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"), NUM.alias("tn"), ADDR.alias("ta"))
    d = (pred.select("s1_id", "cand_id").join(q, on="s1_id").join(r, on="cand_id")
         .filter(pl.col("qn").is_not_null() & (pl.col("qn") == pl.col("tn")))
         .with_columns(add=pl.col("tt").list.set_difference(pl.col("qt")),
                       miss=pl.col("qt").list.set_difference(pl.col("tt")))
         .filter((pl.col("add").list.len() == 1) & (pl.col("miss").list.len() <= 1))
         .with_columns(a=pl.col("add").list.first(), m=pl.col("miss").list.first(), swap=pl.col("miss").list.len()))
    # twin: another S1 carries exactly the record's name tokens at the same number
    twins = q.select(pl.col("qt").list.join(" ").alias("k"), pl.col("qn").alias("tn")).group_by("k", "tn").len("n_twin")
    d = d.with_columns(k=pl.col("tt").list.join(" ")).join(twins, on=["k", "tn"], how="left").with_columns(
        pl.col("n_twin").fill_null(0))
    sim = process.cpdist(d["a"].to_list(), d["m"].fill_null("").to_list(), scorer=fuzz.ratio, workers=-1)
    asim = process.cpdist(d["qa"].to_list(), d["ta"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    d = d.with_columns(sim=pl.Series(sim), asim=pl.Series(asim))
    noise = NOISE[c]
    is_noise = (pl.col("a").is_in(noise) & (pl.col("n_twin") == 0) & (pl.col("asim") >= ASIM_MIN)
                & ((pl.col("swap") == 1) | ~pl.col("a").is_in(SIBLING_ALSO)))
    is_noun = ((pl.col("swap") == 1) & pl.col("a").is_in(vocab) & pl.col("m").is_in(vocab)
               & ~pl.col("a").is_in(noise) & ~pl.col("m").is_in(noise) & (pl.col("sim") < TYPO_MAX)
               & ~pl.col("a").is_in(STOP) & ~pl.col("m").is_in(STOP)
               & (pl.col("asim") >= ASIM_MIN))
    return d.with_columns(kind=pl.when(is_noise).then(pl.lit("noise")).when(is_noun).then(pl.lit("noun"))).filter(
        pl.col("kind").is_not_null()).select("s1_id", "cand_id", "kind")


def apply(pred: pl.DataFrame, sw: pl.DataFrame, which: str) -> pl.DataFrame:
    """Return pred with p overridden for the chosen rule kinds."""
    kinds = ["noun", "noise"] if which == "both" else [which]
    d = pred.join(sw.filter(pl.col("kind").is_in(kinds)), on=["s1_id", "cand_id"], how="left")
    p = (pl.when(pl.col("kind") == "noise").then(pl.max_horizontal(pl.col("p"), pl.lit(P_NOISE)))
         .when(pl.col("kind") == "noun").then(pl.min_horizontal(pl.col("p"), pl.lit(P_NOUN)))
         .otherwise(pl.col("p")))
    return d.with_columns(p.cast(pred["p"].dtype).alias("p")).drop("kind")


def select(pred: pl.DataFrame) -> pl.DataFrame:
    """Production selection: exclusive-owner probabilities, one owner, expected-F0.5 sets."""
    return select_expected_f05(one_owner(exclusive_owner_prob(pred, "p"), "p"), "p").select("s1_id", "cand_id")


def main() -> None:
    mode = sys.argv[1]
    if mode == "eval":
        ev = pl.read_parquet(sys.argv[2])
        gt = load_ground_truth_pairs()
        for c in ("US", "India"):
            o = ev.filter(pl.col("country") == c).select("s1_id", "cand_id", "p", "y")
            sw = swap_pairs("train", c, o)
            s1 = o["s1_id"].unique()
            truth = gt.join(o.select("s1_id").unique(), on="s1_id")
            base = per_entity_f05(truth, select(o.drop("y")), s1)["f05"].mean()
            st = sw.join(o, on=["s1_id", "cand_id"]).group_by("kind").agg(
                n=pl.len(), true=pl.col("y").mean(), mp=pl.col("p").mean())
            print(f"[eval {c}] {st.to_dicts()}", flush=True)
            for which in ("noun", "noise", "both"):
                f = per_entity_f05(truth, select(apply(o.drop("y"), sw, which)), s1)["f05"].mean()
                print(f"[eval {c}] {which:5s}: F0.5 {base:.5f} -> {f:.5f} ({f - base:+.5f})", flush=True)
        return
    src, out = sys.argv[2], sys.argv[3]
    which = sys.argv[4] if len(sys.argv) > 4 else "both"
    fr = pl.scan_parquet(src).filter(pl.col("country") == "France").collect()
    sw = swap_pairs("test", "France", fr)
    n = fr["s1_id"].n_unique()
    st = sw.join(fr, on=["s1_id", "cand_id"]).group_by("kind").agg(n=pl.len(), mp=pl.col("p").mean())
    print(f"France rule pairs: {[(x['kind'], round(x['n'] / n * 1000, 1), round(x['mp'], 3)) for x in st.to_dicts()]}")
    b = select(fr.select("s1_id", "cand_id", "p"))
    fr2 = apply(fr, sw, which)
    a = select(fr2.select("s1_id", "cand_id", "p"))
    rem = b.join(a, on=["s1_id", "cand_id"], how="anti").height
    add = a.join(b, on=["s1_id", "cand_id"], how="anti").height
    print(f"France '{which}': selections removed {rem / n * 1000:.1f}/1k S1, added {add / n * 1000:.1f}/1k S1; "
          f"matches/S1 {b.height / n:.3f} -> {a.height / n:.3f}")
    rest = pl.scan_parquet(src).filter(pl.col("country") != "France")
    pl.concat([rest, fr2.select(fr.columns).lazy()]).sink_parquet(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
