from exar.policy import EffectPolicy, Rules, Strictest
from exar.typing import Action


def test_strictest() -> None:
    policy = Strictest(
        policies=(
            EffectPolicy(write="allow"),
            Rules(by_name={"mail": "deny"}, default="ask"),
        )
    )

    assert policy(Action(kind="tool", name="mail", effect="read")) == "deny"
    assert policy(Action(kind="tool", name="edit", effect="write")) == "ask"
    assert (
        Strictest(policies=())(Action(kind="agent", name="any", effect="execute"))
        == "allow"
    )
