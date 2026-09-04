from congress_nlp.annotation.utils import (
    ordinal,
    chamber_from_type,
    bill_number_display,
    congress_gov_url,
    keyword_count_and_strong,
    is_amendment_type,
)

STRONG = frozenset({"prc", "pla", "uyghur"})


def test_ordinal_common():
    assert ordinal(101) == "101st"
    assert ordinal(102) == "102nd"
    assert ordinal(103) == "103rd"
    assert ordinal(119) == "119th"


def test_ordinal_teens_are_th():
    assert ordinal(111) == "111th"
    assert ordinal(112) == "112th"
    assert ordinal(113) == "113th"


def test_chamber_from_type():
    assert chamber_from_type("hr") == "House"
    assert chamber_from_type("hres") == "House"
    assert chamber_from_type("s") == "Senate"
    assert chamber_from_type("sconres") == "Senate"


def test_bill_number_display():
    assert bill_number_display("hr", "1153") == "H.R. 1153"
    assert bill_number_display("s", "442") == "S. 442"
    assert bill_number_display("sconres", "7") == "S.Con.Res. 7"


def test_congress_gov_url_bill():
    assert congress_gov_url(118, "hr", "1153") == (
        "https://www.congress.gov/bill/118th-congress/house-bill/1153"
    )
    assert congress_gov_url(119, "s", "442") == (
        "https://www.congress.gov/bill/119th-congress/senate-bill/442"
    )


def test_congress_gov_url_resolution():
    assert congress_gov_url(117, "hres", "9") == (
        "https://www.congress.gov/bill/117th-congress/house-resolution/9"
    )


def test_keyword_count_and_strong():
    assert keyword_count_and_strong("tariff", STRONG) == (1, False)
    assert keyword_count_and_strong("china|prc", STRONG) == (2, True)
    assert keyword_count_and_strong("", STRONG) == (0, False)


def test_is_amendment_type():
    assert is_amendment_type("samdt") is True
    assert is_amendment_type("hamdt") is True
    assert is_amendment_type("hr") is False
