"""Gene Ontology: OBO/GAF parsing, DAG structure, similarity, enrichment."""

import math
import textwrap

import numpy as np
import pytest

from nexus.bio.enrichment import (
    benjamini_hochberg,
    bonferroni,
    enrichment,
    hypergeometric_sf,
    log_choose,
)
from nexus.bio.gaf import Annotation, gene_to_terms, parse_gaf, term_to_genes
from nexus.bio.obo import parse_obo
from nexus.bio.ontology import GeneOntology

MINI_OBO = textwrap.dedent("""\
    format-version: 1.2
    data-version: test/2026-01-01

    [Term]
    id: GO:0000001
    name: root process
    namespace: biological_process

    [Term]
    id: GO:0000002
    name: child A
    namespace: biological_process
    is_a: GO:0000001 ! root process

    [Term]
    id: GO:0000003
    name: child B
    namespace: biological_process
    is_a: GO:0000001 ! root process
    alt_id: GO:0000099

    [Term]
    id: GO:0000004
    name: grandchild
    namespace: biological_process
    is_a: GO:0000002 ! child A
    relationship: part_of GO:0000003 ! child B

    [Term]
    id: GO:0000005
    name: obsolete thing
    namespace: biological_process
    is_obsolete: true
    replaced_by: GO:0000004

    [Typedef]
    id: part_of
    name: part of
    """)


@pytest.fixture
def mini_go(tmp_path):
    path = tmp_path / "mini.obo"
    path.write_text(MINI_OBO)
    go = GeneOntology.load(path)
    annotations = [
        Annotation("GENE1", "U1", "GO:0000004", "EXP", "P", False, "taxon:9606"),
        Annotation("GENE2", "U2", "GO:0000004", "EXP", "P", False, "taxon:9606"),
        Annotation("GENE3", "U3", "GO:0000002", "EXP", "P", False, "taxon:9606"),
        Annotation("GENE4", "U4", "GO:0000003", "EXP", "P", False, "taxon:9606"),
        Annotation("GENE5", "U5", "GO:0000001", "EXP", "P", False, "taxon:9606"),
    ]
    go.annotate(annotations, aspect="P")
    return go


# ------------------------------------------------------------------- parsing

def test_obo_parsing(tmp_path):
    path = tmp_path / "mini.obo"
    path.write_text(MINI_OBO)
    terms, header = parse_obo(path)
    assert header["data-version"] == "test/2026-01-01"
    assert len(terms) == 5                      # [Typedef] is skipped
    assert terms["GO:0000002"].name == "child A"
    assert terms["GO:0000002"].is_a == ["GO:0000001"]
    assert terms["GO:0000004"].part_of == ["GO:0000003"]
    assert terms["GO:0000005"].is_obsolete
    assert terms["GO:0000003"].alt_ids == ["GO:0000099"]


def test_gaf_parsing_and_filters(tmp_path):
    rows = [
        "!gaf-version: 2.2",
        "UniProtKB\tU1\tGENE1\tinvolved_in\tGO:0000004\tREF\tEXP\t\tP\tname\tsyn\tprotein\ttaxon:9606\t20240101\tGOC\t\t",
        "UniProtKB\tU2\tGENE2\tNOT|involved_in\tGO:0000004\tREF\tIDA\t\tP\tname\tsyn\tprotein\ttaxon:9606\t20240101\tGOC\t\t",
        "UniProtKB\tU3\tGENE3\tinvolved_in\tGO:0000002\tREF\tIEA\t\tP\tname\tsyn\tprotein\ttaxon:9606\t20240101\tGOC\t\t",
        "UniProtKB\tU4\tGENE4\tinvolved_in\tGO:0000003\tREF\tND\t\tP\tname\tsyn\tprotein\ttaxon:9606\t20240101\tGOC\t\t",
    ]
    path = tmp_path / "a.gaf"
    path.write_text("\n".join(rows) + "\n")
    all_ann = parse_gaf(path)
    assert [a.gene for a in all_ann] == ["GENE1", "GENE3"]      # NOT and ND dropped
    assert [a.gene for a in parse_gaf(path, exclude_electronic=True)] == ["GENE1"]
    assert len(parse_gaf(path, include_negated=True)) == 3
    assert len(parse_gaf(path, taxon="taxon:10090")) == 0


def test_gene_term_maps():
    ann = [
        Annotation("A", "U", "GO:1", "EXP", "P", False, "t"),
        Annotation("A", "U", "GO:2", "EXP", "P", False, "t"),
        Annotation("B", "U", "GO:1", "EXP", "F", False, "t"),
    ]
    assert gene_to_terms(ann)["A"] == {"GO:1", "GO:2"}
    assert term_to_genes(ann)["GO:1"] == {"A", "B"}
    assert term_to_genes(ann, aspect="F")["GO:1"] == {"B"}


# --------------------------------------------------------------- DAG structure

def test_ancestors_follow_is_a_and_part_of(mini_go):
    anc = mini_go.ancestors("GO:0000004")
    assert anc == frozenset({"GO:0000004", "GO:0000002", "GO:0000003", "GO:0000001"})
    assert mini_go.ancestors("GO:0000004", include_self=False) == anc - {"GO:0000004"}
    assert mini_go.ancestors("GO:0000001") == frozenset({"GO:0000001"})


def test_descendants_and_depth(mini_go):
    assert mini_go.descendants("GO:0000001") == {
        "GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"}
    assert mini_go.descendants("GO:0000004") == {"GO:0000004"}


def test_alt_id_and_obsolete_resolution(mini_go):
    assert mini_go.resolve("GO:0000099") == "GO:0000003"
    assert mini_go.resolve("GO:0000005") == "GO:0000004"
    assert mini_go.resolve("GO:9999999") is None


def test_dag_is_acyclic(mini_go):
    order = mini_go.topological_order()
    assert len(order) == len(mini_go)
    position = {t: i for i, t in enumerate(order)}
    for term, parents in mini_go.parents.items():
        for p in parents:
            assert position[term] < position[p]


# ------------------------------------------------------ annotations and content

def test_true_path_rule_propagates_annotations(mini_go):
    # GENE1 and GENE2 are annotated to the grandchild; both must count for
    # every ancestor of it.
    assert mini_go.genes_for("GO:0000004") == {"GENE1", "GENE2"}
    assert mini_go.genes_for("GO:0000002") == {"GENE1", "GENE2", "GENE3"}
    assert mini_go.genes_for("GO:0000001") == {f"GENE{i}" for i in range(1, 6)}


def test_information_content_increases_with_specificity(mini_go):
    ic_root = mini_go.information_content("GO:0000001")
    ic_child = mini_go.information_content("GO:0000002")
    ic_leaf = mini_go.information_content("GO:0000004")
    assert ic_root == pytest.approx(0.0, abs=1e-9)
    assert ic_root < ic_child < ic_leaf
    assert ic_leaf == pytest.approx(-math.log(2 / 5))


# ---------------------------------------------------------- semantic similarity

def test_similarity_identity_and_bounds(mini_go):
    for method in ("lin", "jiang", "wang"):
        assert mini_go.similarity("GO:0000004", "GO:0000004", method) == pytest.approx(1.0)
        v = mini_go.similarity("GO:0000004", "GO:0000003", method)
        assert 0.0 <= v <= 1.0


def test_similarity_is_symmetric(mini_go):
    for method in ("resnik", "lin", "jiang", "wang"):
        a = mini_go.similarity("GO:0000004", "GO:0000002", method)
        b = mini_go.similarity("GO:0000002", "GO:0000004", method)
        assert a == pytest.approx(b)


def test_closer_terms_are_more_similar(mini_go):
    near = mini_go.similarity("GO:0000004", "GO:0000002", "lin")
    far = mini_go.similarity("GO:0000002", "GO:0000003", "lin")
    assert near > far


def test_resnik_equals_ic_of_common_ancestor(mini_go):
    mica, ic = mini_go.mica("GO:0000004", "GO:0000002")
    assert mica == "GO:0000002"
    assert mini_go.similarity("GO:0000004", "GO:0000002", "resnik") == pytest.approx(ic)


def test_gene_similarity_combination_modes(mini_go):
    a, b = ["GO:0000004"], ["GO:0000002", "GO:0000003"]
    for combine in ("bma", "max", "avg"):
        v = mini_go.gene_similarity(a, b, method="lin", combine=combine)
        assert 0.0 <= v <= 1.0
    assert mini_go.gene_similarity(a, [], method="lin") == 0.0


# ------------------------------------------------------------------ statistics

def test_log_choose_matches_exact():
    assert math.exp(log_choose(10, 3)) == pytest.approx(120.0)
    assert log_choose(5, 6) == -math.inf


@pytest.mark.parametrize("k,N,K,n,expected", [
    (0, 100, 10, 10, 1.0),
    (1, 10, 5, 1, 0.5),
    (2, 10, 5, 2, 5 * 4 / (10 * 9)),
    (11, 100, 10, 10, 0.0),
])
def test_hypergeometric_survival(k, N, K, n, expected):
    assert hypergeometric_sf(k, N, K, n) == pytest.approx(expected, abs=1e-12)


def test_hypergeometric_full_distribution_sums_to_one():
    N, K, n = 30, 12, 8
    probs = [hypergeometric_sf(k, N, K, n) - hypergeometric_sf(k + 1, N, K, n)
             for k in range(0, min(K, n) + 1)]
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert all(p >= -1e-12 for p in probs)


def test_benjamini_hochberg_reference_values():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    adj = benjamini_hochberg(p)
    np.testing.assert_allclose(adj[:2], [0.01, 0.04], atol=1e-9)
    assert np.all(np.diff(adj) >= -1e-12)        # monotone
    assert np.all(adj >= p - 1e-12)              # never smaller than raw p
    assert np.all(adj <= 1.0)


def test_bonferroni():
    np.testing.assert_allclose(bonferroni(np.array([0.01, 0.5])), [0.02, 1.0])


# ------------------------------------------------------------------ enrichment

def test_enrichment_recovers_the_seeded_term(mini_go):
    res = enrichment(mini_go, ["GENE1", "GENE2"], min_term_size=1, max_term_size=10)
    assert res, "expected at least one enriched term"
    top = res[0]
    assert top.go_id == "GO:0000004"
    assert top.n_study_hits == 2 and top.n_term_genes == 2
    assert top.fold_enrichment > 1.0
    assert top.p_value <= 1.0


def test_enrichment_empty_study_set(mini_go):
    assert enrichment(mini_go, [], min_term_size=1) == []


def test_enrichment_result_fields_are_consistent(mini_go):
    for r in enrichment(mini_go, ["GENE1", "GENE2", "GENE3"], min_term_size=1):
        assert r.n_study_hits <= min(r.n_study, r.n_term_genes)
        assert r.n_term_genes <= r.n_population
        assert 0.0 <= r.p_value <= 1.0
        assert r.fdr >= r.p_value - 1e-12
        assert len(r.genes) == r.n_study_hits


def test_elim_method_runs_and_reports_subset(mini_go):
    classic = enrichment(mini_go, ["GENE1", "GENE2"], min_term_size=1, method="classic")
    elim = enrichment(mini_go, ["GENE1", "GENE2"], min_term_size=1, method="elim")
    assert {r.go_id for r in elim} <= {r.go_id for r in classic}
