"""Deterministic mock LLM.

Not a toy stand-in: it is grounded in exactly the same structured payload the real prompt is rendered
from, so the whole platform (routing, RAG with citations, approvals, batch, evaluation) is demonstrable
and testable on a laptop or in CI. Quality differences vs. a real model are measured by the
evaluation module rather than hidden.
"""

from __future__ import annotations

import re
import time
from typing import Any

from packages.core.config import get_settings
from packages.core.text import COLOR_NAMES, MATERIAL_NAMES, PRODUCT_TYPES, stem, tokenize
from services.llm.base import LLMProvider, LLMResult, estimate_tokens

SK_TO_CS = {
    "pre": "pro",
    "pri": "při",
    "so": "se",
    "vo": "ve",
    "zo": "ze",
    "aj": "i",
    "alebo": "nebo",
    "ako": "jako",
    "ktorý": "který",
    "ktorá": "která",
    "ktoré": "které",
    "sa": "se",
    "vďaka": "díky",
    "dlhšie": "déle",
    "hodín": "hodin",
    "fľaša": "láhev",
    "fľaše": "láhve",
    "fľašu": "láhev",
    "termofľaša": "termoláhev",
    "hrnček": "hrnek",
    "dáždnik": "deštník",
    "obalom": "obalem",
    "viečkom": "víčkem",
    "viečko": "víčko",
    "ideálne": "ideální",
    "ideálny": "ideální",
    "ideálna": "ideální",
    "každodenné": "každodenní",
    "použitie": "použití",
    "kancelárii": "kanceláři",
    "kanceláriu": "kancelář",
    "cestovanie": "cestování",
    "firemné": "firemní",
    "darčeky": "dárky",
    "darček": "dárek",
    "klientov": "klienty",
    "podujatia": "akce",
    "podujatí": "akcí",
    "konferencie": "konference",
    "veľtrhy": "veletrhy",
    "udržateľnosť": "udržitelnost",
    "dbajú": "dbají",
    "personalizovať": "personalizovat",
    "logom": "logem",
    "vašej": "vaší",
    "potlače": "potisku",
    "potlač": "potisk",
    "tlač": "tisk",
    "gravírovanie": "gravírování",
    "sieťotlač": "sítotisk",
    "farba": "barva",
    "farbe": "barvě",
    "čierna": "černá",
    "biela": "bílá",
    "sivá": "šedá",
    "žltá": "žlutá",
    "prírodná": "přírodní",
    "strieborná": "stříbrná",
    "tmavomodrá": "tmavě modrá",
    "ružová": "růžová",
    "veľkosti": "velikosti",
    "veľkosť": "velikost",
    "nehrdzavejúca": "nerezová",
    "nehrdzavejúcej": "nerezové",
    "oceľ": "ocel",
    "ocele": "oceli",
    "papiera": "papíru",
    "papier": "papír",
    "korok": "korek",
    "drevo": "dřevo",
    "izolácia": "izolace",
    "izolácii": "izolaci",
    "dvojstenná": "dvoustěnná",
    "vákuová": "vakuová",
    "ľahká": "lehká",
    "ľahký": "lehký",
    "použiteľná": "použitelná",
    "použiteľný": "použitelný",
    "spája": "spojuje",
    "dizajn": "design",
    "funkčnosť": "funkčnost",
    "zamestnancov": "zaměstnanců",
    "táto": "tato",
    "výrobok": "výrobek",
    "spracovanie": "zpracování",
    "nabíjanie": "nabíjení",
    "bezdrôtová": "bezdrátová",
    "nabíjačka": "nabíječka",
    "batéria": "baterie",
    "rýchle": "rychlé",
    "udržateľných": "udržitelných",
    "zdrojov": "zdrojů",
    "vetruodolná": "větruodolná",
    "otváranie": "otevírání",
    "uvítacie": "uvítací",
    "kampane": "kampaně",
    "zamerané": "zaměřené",
    "slamy": "slámy",
    "pšeničnej": "pšeničné",
    "šnúrka": "šňůrka",
    "šiltovka": "kšiltovka",
    "polokošeľa": "polokošile",
    "kapucňou": "kapucí",
    "vrecko": "kapsa",
    "vreckom": "kapsou",
    "ideálnym": "ideálním",
    "spoľahlivý": "spolehlivý",
    "spoľahlivá": "spolehlivá",
    "obľúbený": "oblíbený",
    "obľúbená": "oblíbená",
    "firmám": "firmám",
    "zákazníkom": "zákazníkům",
    "vhodným": "vhodným",
    "voľba": "volba",
    "voľbou": "volbou",
    "neobsahuje": "neobsahuje",
    "cestách": "cestách",
    "materiálu": "materiálu",
    "objemom": "objemem",
    "pohodlne": "pohodlně",
    "zmestí": "vejde",
    "väčšiny": "většiny",
    "zabudovaným": "vestavěným",
    "hrubšia": "silnější",
    "príjemný": "příjemný",
    "nosenie": "nošení",
    "ponúka": "nabízí",
}

CATEGORY_SENTENCES = {
    "drinkware": (
        "Praktický spoločník na každodenné použitie v kancelárii aj na cestách.",
        "Praktický společník pro každodenní použití v kanceláři i na cestách.",
    ),
    "apparel": (
        "Príjemný materiál je vhodný na celodenné nosenie pri firemných podujatiach.",
        "Příjemný materiál je vhodný pro celodenní nošení při firemních akcích.",
    ),
    "bags": (
        "Priestranné riešenie na dokumenty, notebook aj osobné veci.",
        "Prostorné řešení pro dokumenty, notebook i osobní věci.",
    ),
    "writing": (
        "Spoľahlivý pomocník pri poznámkach na stretnutiach a konferenciách.",
        "Spolehlivý pomocník při poznámkách na schůzkách a konferencích.",
    ),
    "office": (
        "Užitočný doplnok pracovného stola a obľúbený darček pre zamestnancov.",
        "Užitečný doplněk pracovního stolu a oblíbený dárek pro zaměstnance.",
    ),
    "technology": (
        "Praktický technologický doplnok, ktorý príjemcovia používajú každý deň.",
        "Praktický technologický doplněk, který příjemci používají každý den.",
    ),
    "umbrellas": (
        "Spoľahlivá ochrana pred dažďom pre zamestnancov aj zákazníkov.",
        "Spolehlivá ochrana před deštěm pro zaměstnance i zákazníky.",
    ),
    "lifestyle": (
        "Nenápadný, ale užitočný predmet s dlhou životnosťou.",
        "Nenápadný, ale užitečný předmět s dlouhou životností.",
    ),
}
CATEGORY_USE = {
    "drinkware": (
        "Hodí sa ako firemný darček pre klientov aj ako vybavenie pre zamestnancov.",
        "Hodí se jako firemní dárek pro klienty i jako vybavení pro zaměstnance.",
    ),
    "apparel": (
        "Vhodné pre firemné tímy, promo akcie a jednotný vzhľad na podujatiach.",
        "Vhodné pro firemní týmy, promo akce a jednotný vzhled na akcích.",
    ),
    "bags": (
        "Obľúbená voľba pre konferencie, veľtrhy a uvítacie balíčky.",
        "Oblíbená volba pro konference, veletrhy a uvítací balíčky.",
    ),
    "technology": (
        "Vhodná voľba pre technologické firmy a IT podujatia.",
        "Vhodná volba pro technologické firmy a IT akce.",
    ),
    "office": (
        "Hodí sa na školenia, konferencie aj ako súčasť onboardingu.",
        "Hodí se na školení, konference i jako součást onboardingu.",
    ),
}
TAG_SENTENCES = {
    "insulated": (
        "Vďaka izolácii udrží nápoj dlhšie v požadovanej teplote.",
        "Díky izolaci udrží nápoj déle v požadované teplotě.",
    ),
    "eco": (
        "Vhodná voľba pre firmy, ktoré dbajú na udržateľnosť.",
        "Vhodná volba pro firmy, které dbají na udržitelnost.",
    ),
    "bpa-free": ("Výrobok neobsahuje BPA.", "Výrobek neobsahuje BPA."),
    "leak-proof": ("Uzáver zabraňuje vytečeniu obsahu.", "Uzávěr zabraňuje vytečení obsahu."),
    "usb-c": ("Nabíjanie cez štandardný USB-C konektor.", "Nabíjení přes standardní konektor USB-C."),
    "conference": (
        "Obľúbený produkt na konferencie a veľtrhy.",
        "Oblíbený produkt na konference a veletrhy.",
    ),
    "recycled": (
        "Vyrobené s využitím recyklovaných materiálov.",
        "Vyrobeno s využitím recyklovaných materiálů.",
    ),
    "windproof": ("Konštrukcia odoláva aj silnejšiemu vetru.", "Konstrukce odolá i silnějšímu větru."),
}


def _translate_sk_cs(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        word = m.group(0)
        target = SK_TO_CS.get(word.lower())
        if target is None:
            return word
        return target.capitalize() if word[0].isupper() else target

    return re.sub(r"\w+", repl, text)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]


class MockLLMProvider(LLMProvider):
    name = "mock"
    model = "mock-deterministic-v1"

    def __init__(self, latency_ms_per_token: float = 0.3, base_latency_ms: float = 12.0) -> None:
        self.latency_ms_per_token = latency_ms_per_token
        self.base_latency_ms = base_latency_ms

    def _complete(self, messages, *, task, payload, tools, json_schema, max_tokens, temperature) -> LLMResult:
        handler = getattr(self, f"_task_{task}", self._task_general)
        text = handler(payload, messages)
        in_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        out_tokens = estimate_tokens(text)
        ttft = self.base_latency_ms + in_tokens * 0.01  # prefill cost grows with prompt length
        if self.latency_ms_per_token or self.base_latency_ms:
            time.sleep((ttft + out_tokens * self.latency_ms_per_token) / 1000)
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            ttft_ms=round(ttft, 2),
        )

    # ------------------------------------------------------------------------------------------
    def _task_rag_answer(self, payload: dict[str, Any], _messages) -> str:
        question = payload.get("question", "")
        chunks = payload.get("chunks", [])
        lang = payload.get("language", "en")
        q_terms = {stem(t) for t in tokenize(question)}
        per_chunk: list[list[tuple[float, int, str]]] = []
        for c in chunks:
            ranked = []
            for i, sent in enumerate(_sentences(c["text"])):
                if "[removed:" in sent or len(sent) < 40 or len(sent.split()) < 6:
                    continue  # skip redactions and heading fragments
                overlap = len(q_terms & {stem(t) for t in tokenize(sent)})
                if overlap:
                    # prefer earlier (better reranked) chunks and earlier sentences within a chunk
                    ranked.append((overlap - 0.05 * c["ref"] - 0.01 * i, c["ref"], sent))
            if ranked:
                per_chunk.append(sorted(ranked, key=lambda x: -x[0]))
        if not per_chunk:
            return {
                "sk": "V dokumentoch, ku ktorým máte prístup, som túto informáciu nenašiel.",
                "cs": "V dokumentech, ke kterým máte přístup, jsem tuto informaci nenašel.",
            }.get(lang, "I could not find this information in the documents you are allowed to access.")
        # Coverage first: the best sentence of each source, then second-best sentences.
        per_chunk.sort(key=lambda r: -r[0][0])
        ordered = [r[0] for r in per_chunk] + [x for r in per_chunk for x in r[1:2]]
        picked: list[str] = []
        for _, ref, sent in ordered[:4]:
            picked.append(f"{sent.rstrip('.')}. [{ref}]")
        intro = {"sk": "Podľa interných dokumentov:", "cs": "Podle interních dokumentů:"}.get(
            lang, "According to our internal documents:"
        )
        answer = intro + " " + " ".join(picked)
        flagged = [c["ref"] for c in chunks if c.get("flagged")]
        if flagged:
            refs = ", ".join(f"[{r}]" for r in flagged)
            answer += (
                f"\n\nNote: document {refs} contained an embedded instruction addressed to the assistant. "
                "It was treated as untrusted data and ignored."
            )
        return answer

    def _task_product_description(self, payload: dict[str, Any], _messages) -> str:
        p = payload["product"]
        lang = payload.get("language", "sk")
        idx = 0 if lang == "sk" else 1
        name = (p.get("name_sk") if lang == "sk" else p.get("name_cs") or p.get("name_sk") or "").split(",")[
            0
        ]
        material = MATERIAL_NAMES.get(p.get("material", ""), (p.get("material", ""),) * 2)[idx]
        parts = [
            f"{name} spája praktické použitie s priestorom pre vaše logo."
            if idx == 0
            else f"{name} spojuje praktické použití s prostorem pro vaše logo."
        ]
        cap = f", objem {p['capacity_ml']} ml" if p.get("capacity_ml") else ""
        color = COLOR_NAMES.get(p.get("color", ""), ("", ""))[idx]
        parts.append(
            f"Vyrobené z materiálu {material}{cap}, farba {color}."
            if idx == 0
            else f"Vyrobeno z materiálu {material}{cap}, barva {color}."
        )
        parts.append(CATEGORY_SENTENCES.get(p.get("category", ""), CATEGORY_SENTENCES["lifestyle"])[idx])
        for tag in p.get("tags", []):
            if tag in TAG_SENTENCES:
                parts.append(TAG_SENTENCES[tag][idx])
            if len(parts) >= 5:
                break
        if p.get("category") in CATEGORY_USE:
            parts.append(CATEGORY_USE[p["category"]][idx])
        if p.get("sizes"):
            parts.append(
                ("Dostupné veľkosti: " if idx == 0 else "Dostupné velikosti: ") + ", ".join(p["sizes"]) + "."
            )
        parts.append(
            "Produkt je možné personalizovať logom vašej firmy."
            if idx == 0
            else "Produkt je možné personalizovat logem vaší firmy."
        )
        text = " ".join(parts)
        return text[:600].rsplit(" ", 1)[0] if len(text) > 600 else text

    def _task_translate(self, payload: dict[str, Any], _messages) -> str:
        if payload.get("source") == "sk" and payload.get("target") == "cs":
            return _translate_sk_cs(payload.get("text", ""))
        return payload.get("text", "")

    def _task_classify(self, payload: dict[str, Any], _messages) -> str:
        p = payload.get("product", {})
        for token in tokenize(f"{p.get('name_en', '')} {p.get('name_sk', '')}"):
            if token in PRODUCT_TYPES:
                return f'{{"category": "{PRODUCT_TYPES[token][0]}", "confidence": 0.9}}'
        return '{"category": "lifestyle", "confidence": 0.4}'

    def _task_general(self, payload: dict[str, Any], _messages) -> str:
        lang = payload.get("language", "en")
        if lang == "sk":
            return (
                "Som interný AI asistent NovaPromo. Viem vyhľadávať produkty, odpovedať z interných "
                "dokumentov s citáciami, zistiť sklad a ceny a pripraviť návrhy popisov na schválenie."
            )
        return (
            "I am NovaPromo's internal AI assistant. I can search the product catalogue (hybrid search), "
            "answer questions from internal documents with citations, look up stock and pricing, and draft "
            "product content for human approval."
        )


def build_mock() -> MockLLMProvider:
    s = get_settings()
    if s.app_env == "test":
        return MockLLMProvider(latency_ms_per_token=0.0, base_latency_ms=0.0)
    return MockLLMProvider()
