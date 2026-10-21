"""Citation grounding: exact quotes, near-verbatim quotes, and quotes
attributed to a paper that is not in the run."""
from src.citation_grounding import verify_citations, grounding_summary
from src.models import Citation

SOURCE = (
    "Title: A Paper\n\nAbstract: We show that transformers trained with the new "
    "objective converge twice as fast as the standard baseline on every benchmark."
)


def cite(quote, paper_id="2306.07691"):
    return Citation(quote=quote, paper_id=paper_id, claim="a claim")


def test_exact_quote_is_verified(make_paper):
    paper = make_paper(full_text=SOURCE)
    [v] = verify_citations(
        [cite("transformers trained with the new objective converge twice as fast")],
        {paper.paper_id: paper},
    )
    assert v.verified and v.match_score == 1.0


def test_fuzzy_quote_is_verified(make_paper):
    paper = make_paper(full_text=SOURCE)
    # Same sentence with a few words changed.
    [v] = verify_citations(
        [cite("transformers trained with a new objective converge two times as fast")],
        {paper.paper_id: paper},
    )
    assert v.verified and 0.75 <= v.match_score < 1.0


def test_invented_quote_is_not_verified(make_paper):
    paper = make_paper(full_text=SOURCE)
    [v] = verify_citations(
        [cite("quantum annealing eliminates the need for any training data whatsoever")],
        {paper.paper_id: paper},
    )
    assert not v.verified


def test_unknown_paper_id_is_flagged(make_paper):
    paper = make_paper(full_text=SOURCE)
    [v] = verify_citations([cite("converge twice as fast", paper_id="9999.99999")], {paper.paper_id: paper})
    assert not v.verified and v.match_score == 0.0

    report = grounding_summary([v])
    assert report["hallucinated_citations"] == 1
    assert report["flagged"][0]["paper_id"] == "9999.99999"
