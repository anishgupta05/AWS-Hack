import textwrap

import pytest

from person_b.pomerium_gate import Action, PomeriumGate


@pytest.fixture
def policy_file(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        textwrap.dedent(
            """
            spend_ceiling_usd: 2.00
            allowlist:
              - pull_data
              - zero_enrich
            """
        )
    )
    return p


def test_allowlisted_action_under_ceiling_passes(policy_file):
    gate = PomeriumGate(policy_file)
    decision = gate.check(Action(type="pull_data", cost_usd=0.0))
    assert decision.allowed is True


def test_non_allowlisted_action_denied(policy_file):
    gate = PomeriumGate(policy_file)
    decision = gate.check(Action(type="delete_dataset", cost_usd=0.0))
    assert decision.allowed is False
    assert "allowlist" in decision.reason


def test_action_over_spend_ceiling_denied(policy_file):
    gate = PomeriumGate(policy_file)
    decision = gate.check(Action(type="zero_enrich", cost_usd=5.00))
    assert decision.allowed is False
    assert "ceiling" in decision.reason


def test_spend_accumulates_across_calls(policy_file):
    gate = PomeriumGate(policy_file)
    first = gate.check(Action(type="zero_enrich", cost_usd=1.50))
    second = gate.check(Action(type="zero_enrich", cost_usd=1.00))
    assert first.allowed is True
    assert second.allowed is False  # 1.50 + 1.00 > 2.00 ceiling
