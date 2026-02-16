"""
Shared legality-contract checks for root action handling in MCTS.
"""

from typing import List


def _normalize_action_indices(actions: List[int]) -> List[int]:
    """Normalize action indices to Python ints."""
    return [int(action) for action in actions]


def _find_duplicate_actions(actions: List[int]) -> List[int]:
    """Return sorted duplicate action ids present in `actions`."""
    seen = set()
    duplicates = set()
    for action in actions:
        action_int = int(action)
        if action_int in seen:
            duplicates.add(action_int)
            continue
        seen.add(action_int)
    return sorted(duplicates)


def assert_exact_legal_action_match(
    *,
    expected_actions: List[int],
    observed_actions: List[int],
    context: str,
    contract_name: str = "Legal action contract",
    expected_label: str = "Expected actions",
    observed_label: str = "Observed actions",
) -> List[int]:
    """
    Ensure two legal-action lists match exactly (same order, same elements, no duplicates).
    """
    normalized_expected = _normalize_action_indices(expected_actions)
    normalized_observed = _normalize_action_indices(observed_actions)

    expected_duplicates = _find_duplicate_actions(normalized_expected)
    observed_duplicates = _find_duplicate_actions(normalized_observed)
    if expected_duplicates or observed_duplicates:
        raise ValueError(
            f"{contract_name} violated at {context}. Duplicate action indices are not allowed.\n"
            f"{expected_label} duplicates: {expected_duplicates}\n"
            f"{observed_label} duplicates: {observed_duplicates}\n"
            f"{expected_label} ({len(normalized_expected)}): {normalized_expected}\n"
            f"{observed_label} ({len(normalized_observed)}): {normalized_observed}"
        )

    if normalized_expected != normalized_observed:
        expected_set = set(normalized_expected)
        observed_set = set(normalized_observed)
        missing_in_observed = sorted(expected_set - observed_set)
        unexpected_in_observed = sorted(observed_set - expected_set)
        raise ValueError(
            f"{contract_name} violated at {context}. Expected exact legal-action match.\n"
            f"{expected_label} ({len(normalized_expected)}): {normalized_expected}\n"
            f"{observed_label} ({len(normalized_observed)}): {normalized_observed}\n"
            f"Missing in observed (expected but absent): {missing_in_observed}\n"
            f"Unexpected in observed (present but unexpected): {unexpected_in_observed}"
        )

    return normalized_expected


def assert_actions_subset_of_legal(
    *,
    actions: List[int],
    legal_actions: List[int],
    context: str,
    contract_name: str = "Forced-root legality contract",
    actions_label: str = "Actions",
    legal_label: str = "Legal actions",
) -> List[int]:
    """
    Ensure every action in `actions` is currently legal.

    Duplicates in `actions` are allowed (e.g., repeated forced root allocations).
    """
    normalized_actions = _normalize_action_indices(actions)
    if not normalized_actions:
        return normalized_actions

    normalized_legal_actions = _normalize_action_indices(legal_actions)
    legal_duplicates = _find_duplicate_actions(normalized_legal_actions)
    if legal_duplicates:
        raise ValueError(
            f"{contract_name} violated at {context}. Duplicate legal action indices are not allowed.\n"
            f"{legal_label} duplicates: {legal_duplicates}\n"
            f"{legal_label} ({len(normalized_legal_actions)}): {normalized_legal_actions}"
        )

    legal_set = set(normalized_legal_actions)
    illegal_positions = [
        (idx, action)
        for idx, action in enumerate(normalized_actions)
        if action not in legal_set
    ]
    if illegal_positions:
        illegal_actions = sorted({action for _, action in illegal_positions})
        raise ValueError(
            f"{contract_name} violated at {context}. All actions must be legal at execution time.\n"
            f"Illegal actions: {illegal_actions}\n"
            f"Illegal positions in action list: {illegal_positions}\n"
            f"{actions_label} ({len(normalized_actions)}): {normalized_actions}\n"
            f"{legal_label} ({len(normalized_legal_actions)}): {normalized_legal_actions}"
        )

    return normalized_actions
