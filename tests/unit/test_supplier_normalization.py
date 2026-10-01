import json
import xml.etree.ElementTree as ET

import pytest
from openpyxl import Workbook

from services.ingestion.suppliers import (
    EcoLineXlsxAdapter,
    EuroGiftXmlAdapter,
    IngestionReport,
    PromoTexJsonAdapter,
    canonical_color,
    load_feed,
    parse_decimal,
    parse_volume_ml,
)


def test_parsers():
    assert parse_decimal("12,90") == 12.90
    assert parse_decimal(3) == 3.0
    with pytest.raises(ValueError):
        parse_decimal("N/A")
    assert parse_volume_ml("0,75 L") == 750
    assert parse_volume_ml("750 ml") == 750
    assert parse_volume_ml(None) is None
    assert canonical_color("Schwarz") == "black"
    assert canonical_color("černá") == "black"
    assert canonical_color("Marineblau") == "navy"
    with pytest.raises(ValueError):
        canonical_color("")


def test_xml_adapter(tmp_path):
    xml = """<catalog><item code="11111.01"><master_code>11111</master_code>
      <names><name lang="en">Thermal bottle Test 750 ml, black</name><name lang="sk">Termofľaša Test</name></names>
      <descriptions><description lang="en">Nice.</description></descriptions>
      <category path="Drinkware &gt; Vacuum Flasks &amp; Bottles"/>
      <attributes><attr name="colour">Schwarz</attr><attr name="material">Stainless Steel</attr>
      <attr name="volume">0,75 L</attr></attributes>
      <pricing currency="EUR"><price qty="1">12,90</price></pricing><stock>5</stock><brand>B</brand>
      <origin>CN</origin><leadtime>7</leadtime><keywords>insulated;vacuum</keywords></item></catalog>"""
    path = tmp_path / "s1.xml"
    path.write_text(xml, encoding="utf-8")
    report = IngestionReport()
    load_feed(EuroGiftXmlAdapter(), path, report)
    p = report.products[0]
    assert (p.color, p.material, p.capacity_ml, p.price_eur, p.category) == (
        "black",
        "stainless steel",
        750,
        12.90,
        "drinkware",
    )
    assert p.tags == ["insulated", "vacuum"]
    ET.parse(path)


def test_json_adapter_rejects_negative_price(tmp_path):
    feed = {
        "colour_codes": {"NV": "navy"},
        "products": [
            {
                "id": "PT-1-NV",
                "title": {"en": "T-shirt X, navy", "sk": "Tričko X"},
                "body": {"en": "b"},
                "group": "T-Shirts",
                "colour_code": "NV",
                "fabric": "100% organic cotton, 160 g/m2",
                "sizes": [{"size": "M", "stock": 3}, {"size": "XL", "stock": 4}],
                "price": {"amount_cents": 499},
                "delivery_days": 5,
                "labels": [],
            },
            {
                "id": "PT-2-NV",
                "title": {"en": "Bad", "sk": "Bad"},
                "group": "T-Shirts",
                "colour_code": "NV",
                "fabric": "100% cotton",
                "sizes": [],
                "price": {"amount_cents": -1},
            },
        ],
    }
    path = tmp_path / "s2.json"
    path.write_text(json.dumps(feed), encoding="utf-8")
    report = IngestionReport()
    load_feed(PromoTexJsonAdapter(), path, report)
    assert len(report.products) == 1 and len(report.rejected) == 1
    p = report.products[0]
    assert (p.material, p.sizes, p.stock, p.price_eur) == ("organic cotton", ["M", "XL"], 7, 4.99)
    assert "non-positive price" in report.rejected[0]["reason"]


def test_xlsx_adapter_converts_czk_and_czech_values(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Produkty"
    ws.append(list(EcoLineXlsxAdapter.COLUMNS))
    ws.append(
        [
            "ECO-1",
            "Zápisník",
            "Zápisník SK",
            "Notebook",
            "Popis",
            "",
            "Kancelář / Zápisníky",
            "přírodní",
            "bambus",
            "",
            250,
            10,
            12,
            "EcoLine",
            "CZ",
            "eco, bamboo",
        ]
    )
    ws.append(
        ["", "No code", "", "", "", "", "Kancelář / Zápisníky", "černá", "bambus", "", 100, 1, 1, "", "", ""]
    )
    path = tmp_path / "s3.xlsx"
    wb.save(path)
    report = IngestionReport()
    load_feed(EcoLineXlsxAdapter(), path, report)
    p = report.products[0]
    assert (p.price_eur, p.color, p.material, p.category, p.tags) == (
        10.0,
        "natural",
        "bamboo",
        "office",
        ["eco", "bamboo"],
    )
    assert report.rejected[0]["source"].endswith("row 3")


def test_real_feeds_normalise_with_expected_rejections(platform):
    report = platform["report"]
    assert len(report.products) > 1000
    assert len(report.rejected) == 7
    assert {r["supplier"] for r in report.rejected} == {"EGT", "PTX", "ECO"}
    assert all(p.price_eur > 0 and p.color for p in report.products)
