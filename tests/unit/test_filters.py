import pytest

from services.search.filters import extract_query


def test_scenario_query_extracts_all_constraints():
    q = extract_query("Find black metal bottles around 750 ml under 15 EUR that are in stock.")
    f = q.filters
    assert f.color == "black"
    assert f.material == "metal"
    assert f.category == "drinkware"
    assert f.price_max == 15
    assert f.in_stock is True
    assert f.capacity_min_ml < 750 < f.capacity_max_ml
    assert "in stock" not in q.semantic_text and "15" not in q.semantic_text
    assert "bottles" in q.semantic_text


@pytest.mark.parametrize(
    "text,expected_max",
    [
        ("gift for conference attendees under 8 EUR", 8.0),
        ("pens below 2,50 €", 2.5),
        ("hľadám čiernu termofľašu do 15 eur", 15.0),
        ("mug max 5 euro", 5.0),
    ],
)
def test_price_max_variants(text, expected_max):
    assert extract_query(text).filters.price_max == expected_max


def test_price_range_and_around():
    assert (
        extract_query("bags between 5 and 10 EUR").filters.price_min,
        extract_query("bags between 5 and 10 EUR").filters.price_max,
    ) == (5, 10)
    around = extract_query("umbrella around 20 EUR").filters
    assert around.price_min == 16 and around.price_max == 24


def test_capacity_in_litres_is_not_a_price():
    f = extract_query("bottle under 1 l").filters
    assert f.price_max is None
    assert f.capacity_min_ml == 900 and f.capacity_max_ml == 1100


def test_size_colour_material_textile():
    f = extract_query("blue cotton t-shirt size XL").filters
    assert (f.color, f.material, f.size, f.category) == ("blue", "cotton", "XL", "apparel")


def test_slovak_and_czech_vocabulary():
    sk = extract_query("modré bavlnené tričko skladom")
    assert sk.filters.color == "blue" and sk.filters.material == "cotton" and sk.filters.in_stock
    assert sk.detected_language == "sk"
    cs = extract_query("černá termoláhev skladem")
    assert cs.filters.color == "black" and cs.filters.category == "drinkware" and cs.filters.in_stock


@pytest.mark.parametrize(
    "text,code",
    [
        ("product 29904.01", "29904.01"),
        ("NP-10042", "NP-10042"),
        ("stock of np-10042?", "NP-10042"),
        ("PT-4410-NV details", "PT-4410-NV"),
    ],
)
def test_exact_codes(text, code):
    assert extract_query(text).exact_code == code


def test_no_false_filters_for_plain_semantic_query():
    f = extract_query("eco friendly gifts for IT company").filters
    assert f.active() == {"language": "en"}
