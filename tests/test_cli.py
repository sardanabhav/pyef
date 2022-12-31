"""Tests for the `cli` module."""


from pyef import cli


def test_main() -> None:
    """Basic CLI test."""
    assert cli.main([]) == 0


# def test_show_help(capsys: pytest.fixture[...]) -> None:
#     """
#     Show help.

#     Arguments:
#         capsys: Pytest fixture to capture output.
#     """
#     with pytest.raises(SystemExit):
#         cli.main(["-h"])
#     captured = capsys.readouterr()
#     assert "pyef" in captured.out
