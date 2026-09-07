import felupe_contact
from felupe_contact import SolidBodyContact
from felupe_contact.__about__ import __version__


def test_import():
    "The package exports the contact class."

    assert felupe_contact.__all__ == ["SolidBodyContact"]
    assert felupe_contact.SolidBodyContact is SolidBodyContact


def test_version():
    "The version is a non-empty string."

    assert isinstance(__version__, str)
    assert len(__version__) > 0


if __name__ == "__main__":
    test_import()
    test_version()
