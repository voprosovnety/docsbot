from bot.rag import _is_bare_refusal


class TestBareRefusalDetection:
    def test_plain_form(self):
        assert _is_bare_refusal("I don't know")

    def test_with_trailing_period(self):
        assert _is_bare_refusal("I don't know.")

    def test_case_insensitive(self):
        assert _is_bare_refusal("i DON'T know")

    def test_curly_apostrophe(self):
        assert _is_bare_refusal("I don’t know")

    def test_missing_apostrophe(self):
        assert _is_bare_refusal("I dont know")

    def test_surrounding_whitespace(self):
        assert _is_bare_refusal("  I don't know  \n")

    def test_real_answer_is_not_a_refusal(self):
        assert not _is_bare_refusal(
            "Full-time staff get 28 days.\n\nSource: handbook.md"
        )

    def test_answer_that_merely_mentions_uncertainty_is_kept(self):
        # The model qualifying part of a real answer must not be swallowed.
        assert not _is_bare_refusal(
            "The policy covers 28 days; I don't know whether it changed in 2026."
        )
