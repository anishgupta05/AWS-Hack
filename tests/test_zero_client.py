import textwrap

import pytest

from person_b.pomerium_gate import PomeriumGate
from person_b.zero_client import ServiceListing, ZeroClient

pytestmark = pytest.mark.asyncio


@pytest.fixture
def gate(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        textwrap.dedent(
            """
            spend_ceiling_usd: 10.00
            allowlist:
              - zero_discover
              - zero_enrich
            """
        )
    )
    return PomeriumGate(p)


async def test_discover_returns_listings(gate):
    client = ZeroClient(gate)
    # No real `zero` CLI / capability wired up in test env -> falls back to
    # the mock catalog, which is exactly what we're asserting still works.
    listings = await client.discover("cardiac clinical enrichment")
    assert len(listings) > 0
    assert all(isinstance(listing, ServiceListing) for listing in listings)


async def test_select_and_pay_picks_best_fit_not_first(gate):
    client = ZeroClient(gate)
    low_fit = ServiceListing(
        service_id="low", name="Generic Forum Bot", price_usd=0.1,
        availability="unknown", fit_score=0.5,
    )
    high_fit = ServiceListing(
        service_id="high", name="Cardiac Clinical Data Enricher", price_usd=0.1,
        availability="healthy", fit_score=25.0,
    )
    chosen = await client.select_and_pay([low_fit, high_fit])
    assert chosen.service_id == "high"


async def test_select_and_pay_denied_over_spend_ceiling(gate):
    client = ZeroClient(gate)
    expensive = ServiceListing(
        service_id="pricey", name="Pricey", price_usd=20.0,
        availability="healthy", fit_score=10.0,
    )
    with pytest.raises(PermissionError):
        await client.select_and_pay([expensive])


async def test_select_and_pay_empty_raises(gate):
    client = ZeroClient(gate)
    with pytest.raises(ValueError):
        await client.select_and_pay([])


async def test_select_and_pay_skips_fetch_without_capability_url(gate):
    client = ZeroClient(gate)
    no_url = ServiceListing(
        service_id="mockish", name="Mock Listing", price_usd=0.1,
        availability="healthy", fit_score=5.0, capability_url="",
    )
    chosen = await client.select_and_pay([no_url])
    assert chosen.service_id == "mockish"


async def test_fit_score_prefers_task_relevant_text_over_generic():
    from person_b.zero_client import _task_fit_score

    cardiac = _task_fit_score("Cardiac Clinical Data Enricher for heart disease patients", "healthy", 0.5, "cardiac clinical enrichment")
    generic = _task_fit_score("Generic Forum Scraper", "healthy", 0.5, "cardiac clinical enrichment")
    assert cardiac > generic


async def test_parse_cost_handles_real_cost_object():
    from person_b.zero_client import _parse_cost

    assert _parse_cost({"cost": {"amount": "0.12", "asset": "USDC"}}) == pytest.approx(0.12)
    assert _parse_cost({"cost": 0.5}) == 0.5
    assert _parse_cost({}) == 0.0


async def test_parse_listing_maps_real_search_shape():
    from person_b.zero_client import _parse_listing

    item = {
        "token": "z_9xRsEW.7",
        "canonicalName": "CMS Hospital Care Compare Quality Scores",
        "description": "CMS Hospital Compare quality scores by hospital.",
        "whatItDoes": "Returns safety, readmissions, mortality, star rating.",
        "method": "GET",
        "url": "https://www.stratalize.com/api/x402/hospital-care-compare-quality",
        "cost": {"amount": "0.02", "asset": "USDC"},
        "rating": {"score": "0.00", "reviews": 0, "state": "unrated"},
        "availabilityStatus": "unknown",
    }
    listing = _parse_listing(item, "cardiac clinical enrichment")
    assert listing.service_id == "z_9xRsEW.7"
    assert listing.name == "CMS Hospital Care Compare Quality Scores"
    assert listing.price_usd == pytest.approx(0.02)
    assert listing.method == "GET"
    assert listing.capability_url == item["url"]
    assert listing.rating_state == "unrated"


async def test_extract_field_candidates_handles_zod_invalid_type():
    from person_b.zero_client import _extract_field_candidates

    result = {"body": {"error": {"details": {"issues": [
        {"code": "invalid_type", "path": ["hospital_name"]},
    ]}}}}
    assert "hospital_name" in _extract_field_candidates(result)


async def test_extract_field_candidates_handles_custom_free_text():
    from person_b.zero_client import _extract_field_candidates

    result = {"body": {"error": {"details": {"issues": [
        {"code": "custom", "path": [], "message": "Provide at least one of facilityId, state, city, or name."},
    ]}}}}
    candidates = _extract_field_candidates(result)
    assert "facilityId" in candidates
    assert "city" in candidates


async def test_extract_field_candidates_handles_flat_param_example_shape():
    """Real shape from the Stratalize CMS Hospital Care Compare capability:
    {"error": "invalid_argument", "param": "hospital_name",
     "example": {"hospital_name": "example"}, "message": "Required"}
    -- confirmed live 2026-07-17, this is the exact shape that let the
    self-heal retry succeed and return real signed CMS data."""
    from person_b.zero_client import _extract_field_candidates, _build_retry_params

    result = {"body": {
        "error": "invalid_argument",
        "param": "hospital_name",
        "valid_values": [],
        "example": {"hospital_name": "example"},
        "message": "Required",
    }}
    candidates = _extract_field_candidates(result)
    assert "hospital_name" in candidates
    retry_params = _build_retry_params(candidates)
    assert retry_params == {"hospital_name": "Cleveland Clinic"}


async def test_extract_field_candidates_handles_bare_string_error():
    from person_b.zero_client import _extract_field_candidates

    result = {"body": {"error": "Use POST to execute paid tool calls"}}
    candidates = _extract_field_candidates(result)
    assert "POST" in candidates


async def test_build_retry_params_returns_none_for_unknown_fields():
    from person_b.zero_client import _build_retry_params

    assert _build_retry_params(["some_totally_unknown_field"]) is None
