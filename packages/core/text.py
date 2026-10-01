"""Multilingual (Slovak / Czech / English) text normalisation and domain lexicons.

Slovak and Czech are highly inflected ("fľaša, fľaše, fľašu, fľašou"), so plain token equality
performs badly. We normalise diacritics, apply a light suffix stemmer and map domain vocabulary
(colours, materials, product types) to canonical English values used in SQL and search filters.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

STOPWORDS = {
    # en
    "a",
    "an",
    "the",
    "for",
    "of",
    "and",
    "or",
    "with",
    "in",
    "on",
    "to",
    "is",
    "are",
    "i",
    "we",
    "our",
    "need",
    "want",
    "find",
    "show",
    "me",
    "some",
    "that",
    "this",
    "it",
    "be",
    "by",
    "as",
    "at",
    "from",
    "what",
    "which",
    "how",
    "do",
    "does",
    "any",
    "please",
    "can",
    "you",
    "product",
    "products",
    "item",
    "items",
    "about",
    "per",
    # sk / cs
    "aj",
    "alebo",
    "nebo",
    "pre",
    "pro",
    "na",
    "v",
    "vo",
    "ve",
    "s",
    "so",
    "se",
    "z",
    "zo",
    "ze",
    "od",
    "je",
    "su",
    "jsou",
    "ktory",
    "ktora",
    "ktere",
    "ake",
    "jake",
    "mi",
    "nam",
    "potrebujem",
    "potrebuji",
    "hladam",
    "hledam",
    "chcem",
    "chci",
    "nejake",
    "produkt",
    "produkty",
    "co",
}

COLORS: dict[str, list[str]] = {
    "black": [
        "black",
        "schwarz",
        "cierna",
        "cierny",
        "cierne",
        "ciernej",
        "ciernu",
        "cerna",
        "cerny",
        "cerne",
        "cernou",
    ],
    "white": ["white", "weiss", "biela", "biely", "biele", "bila", "bily", "bile"],
    "blue": ["blue", "blau", "modra", "modry", "modre", "royal"],
    "navy": ["navy", "marineblau", "tmavomodra", "tmavomodry", "namornicka"],
    "red": ["red", "rot", "cervena", "cerveny", "cervene"],
    "green": ["green", "grun", "zelena", "zeleny", "zelene"],
    "grey": ["grey", "grau", "gray", "siva", "sivy", "sive", "seda", "sedy", "sede"],
    "yellow": ["yellow", "gelb", "zlta", "zlty", "zluta", "zluty"],
    "orange": ["orange", "oranzova", "oranzovy"],
    "natural": ["natural", "beige", "prirodna", "prirodni", "bezova"],
    "silver": ["silver", "silber", "strieborna", "stribrna"],
    "pink": ["pink", "ruzova", "ruzovy"],
}
COLOR_LOOKUP = {syn: canon for canon, syns in COLORS.items() for syn in syns}

COLOR_NAMES = {
    "black": ("čierna", "černá"),
    "white": ("biela", "bílá"),
    "blue": ("modrá", "modrá"),
    "navy": ("tmavomodrá", "tmavě modrá"),
    "red": ("červená", "červená"),
    "green": ("zelená", "zelená"),
    "grey": ("sivá", "šedá"),
    "yellow": ("žltá", "žlutá"),
    "orange": ("oranžová", "oranžová"),
    "natural": ("prírodná", "přírodní"),
    "silver": ("strieborná", "stříbrná"),
    "pink": ("ružová", "růžová"),
}

# Material family -> concrete material values stored in SQL.
MATERIAL_FAMILIES: dict[str, list[str]] = {
    "metal": ["stainless steel", "aluminium"],
    "stainless steel": ["stainless steel"],
    "aluminium": ["aluminium"],
    "cotton": ["cotton", "organic cotton"],
    "organic cotton": ["organic cotton"],
    "plastic": ["plastic", "abs"],
    "bamboo": ["bamboo"],
    "cork": ["cork"],
    "glass": ["glass"],
    "ceramic": ["ceramic"],
    "paper": ["recycled paper"],
    "polyester": ["polyester", "recycled pet"],
    "recycled pet": ["recycled pet"],
    "wood": ["wood"],
    "jute": ["jute"],
}
MATERIAL_SYNONYMS: dict[str, str] = {
    "metal": "metal",
    "metallic": "metal",
    "kov": "metal",
    "kovova": "metal",
    "kovovy": "metal",
    "kovove": "metal",
    "steel": "stainless steel",
    "stainless": "stainless steel",
    "nerez": "stainless steel",
    "nerezova": "stainless steel",
    "nerezovy": "stainless steel",
    "ocel": "stainless steel",
    "aluminium": "aluminium",
    "aluminum": "aluminium",
    "hlinik": "aluminium",
    "hlinikova": "aluminium",
    "cotton": "cotton",
    "bavlna": "cotton",
    "bavlnene": "cotton",
    "bavlneny": "cotton",
    "bavlnena": "cotton",
    "plastic": "plastic",
    "plast": "plastic",
    "plastova": "plastic",
    "plastovy": "plastic",
    "bamboo": "bamboo",
    "bambus": "bamboo",
    "bambusova": "bamboo",
    "bambusovy": "bamboo",
    "cork": "cork",
    "korok": "cork",
    "korek": "cork",
    "korkova": "cork",
    "glass": "glass",
    "sklo": "glass",
    "sklenena": "glass",
    "sklenany": "glass",
    "ceramic": "ceramic",
    "keramika": "ceramic",
    "keramicky": "ceramic",
    "keramicka": "ceramic",
    "polyester": "polyester",
    "rpet": "recycled pet",
    "wood": "wood",
    "wooden": "wood",
    "drevo": "wood",
    "drevena": "wood",
    "dreveny": "wood",
    "jute": "jute",
    "juta": "jute",
    "jutova": "jute",
}

MATERIAL_NAMES = {
    "stainless steel": ("nehrdzavejúca oceľ", "nerezová ocel"),
    "aluminium": ("hliník", "hliník"),
    "cotton": ("bavlna", "bavlna"),
    "organic cotton": ("organická bavlna", "organická bavlna"),
    "polyester": ("polyester", "polyester"),
    "recycled pet": ("recyklovaný PET", "recyklovaný PET"),
    "bamboo": ("bambus", "bambus"),
    "cork": ("korok", "korek"),
    "glass": ("sklo", "sklo"),
    "ceramic": ("keramika", "keramika"),
    "recycled paper": ("recyklovaný papier", "recyklovaný papír"),
    "plastic": ("plast", "plast"),
    "abs": ("ABS plast", "ABS plast"),
    "wheat straw": ("pšeničná slama", "pšeničná sláma"),
    "wood": ("drevo", "dřevo"),
    "jute": ("juta", "juta"),
}

# Product-type vocabulary -> (category, subcategory)
PRODUCT_TYPES: dict[str, tuple[str, str]] = {
    "bottle": ("drinkware", "bottles"),
    "bottles": ("drinkware", "bottles"),
    "flask": ("drinkware", "bottles"),
    "thermos": ("drinkware", "bottles"),
    "flasa": ("drinkware", "bottles"),
    "flase": ("drinkware", "bottles"),
    "flasu": ("drinkware", "bottles"),
    "flasky": ("drinkware", "bottles"),
    "termoflasa": ("drinkware", "bottles"),
    "lahev": ("drinkware", "bottles"),
    "termolahev": ("drinkware", "bottles"),
    "lahve": ("drinkware", "bottles"),
    "termoska": ("drinkware", "bottles"),
    "mug": ("drinkware", "mugs"),
    "mugs": ("drinkware", "mugs"),
    "cup": ("drinkware", "mugs"),
    "hrncek": ("drinkware", "mugs"),
    "hrnek": ("drinkware", "mugs"),
    "salka": ("drinkware", "mugs"),
    "tumbler": ("drinkware", "tumblers"),
    "t-shirt": ("apparel", "t-shirts"),
    "tshirt": ("apparel", "t-shirts"),
    "t-shirts": ("apparel", "t-shirts"),
    "tee": ("apparel", "t-shirts"),
    "tricko": ("apparel", "t-shirts"),
    "tricka": ("apparel", "t-shirts"),
    "triko": ("apparel", "t-shirts"),
    "polo": ("apparel", "polo shirts"),
    "hoodie": ("apparel", "hoodies"),
    "mikina": ("apparel", "hoodies"),
    "cap": ("apparel", "caps"),
    "siltovka": ("apparel", "caps"),
    "ksiltovka": ("apparel", "caps"),
    "backpack": ("bags", "backpacks"),
    "batoh": ("bags", "backpacks"),
    "ruksak": ("bags", "backpacks"),
    "tote": ("bags", "tote bags"),
    "taska": ("bags", "tote bags"),
    "bag": ("bags", "tote bags"),
    "pen": ("writing", "pens"),
    "pens": ("writing", "pens"),
    "pero": ("writing", "pens"),
    "propiska": ("writing", "pens"),
    "notebook": ("office", "notebooks"),
    "zapisnik": ("office", "notebooks"),
    "blok": ("office", "notebooks"),
    "powerbank": ("technology", "power banks"),
    "power-bank": ("technology", "power banks"),
    "powerbanka": ("technology", "power banks"),
    "umbrella": ("umbrellas", "umbrellas"),
    "umbrellas": ("umbrellas", "umbrellas"),
    "dazdnik": ("umbrellas", "umbrellas"),
    "destnik": ("umbrellas", "umbrellas"),
    "speaker": ("technology", "speakers"),
    "reproduktor": ("technology", "speakers"),
    "lanyard": ("lifestyle", "lanyards"),
    "lanyards": ("lifestyle", "lanyards"),
}

# Concept lexicon: lets the offline embedder capture *intent* (eco, conference, tech ...) rather
# than only surface tokens. A neural multilingual embedder learns this; here it is explicit.
CONCEPTS: dict[str, list[str]] = {
    "eco": [
        "eco",
        "eco-friendly",
        "ecological",
        "sustainable",
        "sustainability",
        "green",
        "recycled",
        "recyclable",
        "organic",
        "bamboo",
        "cork",
        "jute",
        "rpet",
        "biodegradable",
        "planet",
        "zero-waste",
        "ekologicky",
        "ekologicke",
        "ekologicka",
        "udrzatelny",
        "udrzatelne",
        "recyklovany",
        "recyklovane",
        "wheat",
        "straw",
        "fsc",
        "environment",
        "environmental",
        "plastic-free",
        "natural",
    ],
    "event": [
        "conference",
        "conferences",
        "event",
        "events",
        "attendees",
        "attendee",
        "trade",
        "fair",
        "expo",
        "summit",
        "meetup",
        "congress",
        "delegates",
        "participants",
        "konferencia",
        "konference",
        "vystava",
        "veltrh",
        "podujatie",
        "akcia",
        "welcome",
        "giveaway",
        "giveaways",
        "lanyard",
        "badge",
    ],
    "tech": [
        "technology",
        "tech",
        "it",
        "software",
        "developer",
        "developers",
        "digital",
        "startup",
        "usb",
        "wireless",
        "charger",
        "power",
        "bank",
        "powerbank",
        "cable",
        "speaker",
        "bluetooth",
        "webcam",
        "gadget",
        "gadgets",
        "technologicky",
        "technologicka",
        "hackathon",
        "engineers",
    ],
    "drink": [
        "drink",
        "drinking",
        "bottle",
        "flask",
        "thermos",
        "thermal",
        "insulated",
        "vacuum",
        "mug",
        "cup",
        "tumbler",
        "coffee",
        "tea",
        "water",
        "hydration",
        "flasa",
        "lahev",
        "hrncek",
        "napoj",
    ],
    "office": [
        "office",
        "desk",
        "notebook",
        "pen",
        "writing",
        "stationery",
        "planner",
        "notes",
        "kancelaria",
        "kancelar",
        "zapisnik",
        "pero",
    ],
    "outdoor": [
        "outdoor",
        "hiking",
        "travel",
        "sport",
        "sports",
        "rain",
        "weather",
        "umbrella",
        "camping",
        "backpack",
        "turistika",
        "cestovanie",
    ],
    "apparel": [
        "shirt",
        "t-shirt",
        "tshirt",
        "polo",
        "hoodie",
        "cap",
        "apparel",
        "textile",
        "clothing",
        "wear",
        "uniform",
        "tricko",
        "mikina",
        "oblecenie",
    ],
    "budget": [
        "cheap",
        "budget",
        "affordable",
        "inexpensive",
        "low-cost",
        "giveaway",
        "giveaways",
        "lacny",
        "lacne",
        "levny",
    ],
    "premium": [
        "premium",
        "luxury",
        "executive",
        "vip",
        "elegant",
        "exclusive",
        "gift",
        "gifts",
        "darcek",
        "darceky",
        "prestige",
    ],
}
CONCEPT_LOOKUP = {w: c for c, words in CONCEPTS.items() for w in words}

SK_MARKERS = {"ä", "ľ", "ĺ", "ŕ", "ô"}
CS_MARKERS = {"ř", "ů", "ě"}
SK_WORDS = {
    "pre",
    "sa",
    "ako",
    "som",
    "je",
    "aky",
    "aká",
    "ktoré",
    "nám",
    "potrebujem",
    "hľadám",
    "skladom",
    "koľko",
    "čo",
    "dodávky",
    "našu",
    "naša",
    "prosím",
    "popis",
    "slovenský",
}
CS_WORDS = {
    "pro",
    "jak",
    "jsem",
    "jaký",
    "která",
    "potřebuji",
    "hledám",
    "skladem",
    "kolik",
    "co",
    "dodávky",
    "naše",
    "prosím",
    "český",
    "zboží",
}


def strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    return strip_diacritics(text.lower())


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(normalize(text)):
        tokens.append(tok)
        if "-" in tok and not any(ch.isdigit() for ch in tok):  # "eco-friendly" -> + "eco", "friendly"
            tokens.extend(p for p in tok.split("-") if p)
    return tokens if keep_stopwords else [t for t in tokens if t not in STOPWORDS]


# Lexical query expansion (the in-process equivalent of an Elasticsearch synonym_graph filter).
# Only a few high-precision expansions per concept: expansion widens recall, the reranker restores precision.
QUERY_EXPANSIONS: dict[str, list[str]] = {
    "eco": ["recycled", "bamboo", "organic", "cork", "jute", "sustainable"],
    "event": ["conference", "event", "lanyard"],
    "tech": ["tech", "usb-c", "wireless"],
    "budget": ["giveaway"],
}


def expand_query(text: str) -> str:
    extra: list[str] = []
    for tok in tokenize(text):
        concept = CONCEPT_LOOKUP.get(tok)
        if concept in QUERY_EXPANSIONS:
            extra += [w for w in QUERY_EXPANSIONS[concept] if w not in extra]
    return f"{text} {' '.join(extra)}" if extra else text


_SUFFIXES = ("ami", "ach", "ov", "om", "ou", "ej", "es", "ie", "a", "e", "u", "y", "i", "o", "s")


def stem(token: str) -> str:
    """Very light suffix stripping good enough for SK/CS/EN inflection in short product text."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[: -len(suf)]
    return token


_TYPE_STEMS: dict[str, tuple[str, str]] = {}


def product_type(token: str) -> tuple[str, str] | None:
    """Look up a product type, tolerant to inflection ("termofľašu" -> termofľaša -> drinkware/bottles)."""
    if token in PRODUCT_TYPES:
        return PRODUCT_TYPES[token]
    if not _TYPE_STEMS:
        _TYPE_STEMS.update({stem(k): v for k, v in PRODUCT_TYPES.items() if len(k) > 4})
    return _TYPE_STEMS.get(stem(token))


def detect_language(text: str) -> str:
    lower = text.lower()
    if any(c in lower for c in SK_MARKERS):
        return "sk"
    if any(c in lower for c in CS_MARKERS):
        return "cs"
    words = set(re.findall(r"\w+", lower))
    sk, cs = len(words & SK_WORDS), len(words & CS_WORDS)
    if sk > cs:
        return "sk"
    if cs > sk:
        return "cs"
    if re.search(r"[áéíóúýčďňšťž]", lower):
        return "sk"
    return "en"
