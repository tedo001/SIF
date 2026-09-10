"""Stage 6c - generate written safety information.

Everything upstream turns a report into structured facts. This turns those
facts back into prose - the last mile of "read, analyse, and generate written
safety information" that a human actually circulates: a one-paragraph brief on
a single report, or a multi-section bulletin over a whole corpus.

Deliberately templated, not model-generated. A safety bulletin is exactly the
wrong place for a language model to improvise: every sentence here is built
from fields the pipeline already extracted and verified, so what it says can be
traced back to a specific report and a specific rule the way everything else in
this system can be audited. The local LLM (:mod:`sif.llm`) reads and
translates; it does not write the record.

Two entry points:

:func:`report_brief`
    One paragraph on a single :class:`~sif.pipeline.PipelineResult` - what was
    found, why it matters, what to do. This is what "explanation" text becomes
    when a person needs to hand it to someone else rather than read it off a
    screen.
:func:`corpus_bulletin`
    A multi-section bulletin over a whole analysed corpus: headline numbers,
    what is driving risk, repeat exposures, and the specific reports still
    awaiting a person - the thing an HSE lead would actually circulate at the
    end of a shift or a week.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import Intelligence, PipelineResult

__all__ = ["report_brief", "corpus_bulletin"]

#: Every generated document carries this - the console's own description of
#: itself, not a claim about what the reader should do with the output.
_DISCLAIMER = "Prototype output - for review, not a statutory record."

#: How many outstanding items the bulletin names individually before it falls
#: back to a count. A bulletin that lists forty rows is not read.
_MAX_LISTED_ITEMS = 10


def report_brief(result: "PipelineResult") -> str:
    """One paragraph on a single report, for handing to someone else.

    Built entirely from fields the pipeline already populated - nothing here is
    inferred fresh, so the brief can never say something the structured record
    does not already support.
    """
    reference = result.reference or "This report"
    if not result.iogp_rule:
        return f"{reference}: no text was extracted - nothing to summarise."

    if result.sif_potential:
        lines = [
            f"{reference} carries **fatal potential** under {result.iogp_rule} "
            f"(risk {result.risk_score:.0f}/100, {result.risk_band} band): "
            f"{result.energy_source or 'a high-energy source'} was uncontrolled "
            f"because {(result.barrier_failure or 'a critical barrier').lower()}."
        ]
        if result.minimizing_language:
            lines.append(
                "The report's own wording downplays this - the finding rests on "
                "the extracted facts, not the tone.")
    else:
        lines = [f"{reference}: classified under {result.iogp_rule}, "
                 f"risk {result.risk_score:.0f}/100 ({result.risk_band} band)."]
        if result.high_energy and not result.barrier_failed:
            lines.append(
                f"{result.energy_source or 'A high-energy source'} was present with "
                "no failed barrier recognised - controlled work, or a barrier "
                "described in words the system has not yet learned.")

    if result.needs_review:
        lines.append(f"Queued for review: {result.review_reason}")
    if result.source_language:
        lines.append(f"Translated from {result.source_language} for analysis; the "
                     "original wording is kept as the record.")
    return " ".join(lines)


def corpus_bulletin(intelligence: "Intelligence", results: Sequence["PipelineResult"],
                    title: str = "Safety Intelligence Bulletin", period: str = "",
                    generated_at: str = "") -> str:
    """A multi-section bulletin over one analysed corpus.

    ``period`` is a free-text label for the reporting window ("Week ending
    2026-09-13", "Shift A, 10 Sep"); left blank it is omitted. ``generated_at``
    defaults to now, and is only a parameter so a test can pin it.
    """
    kpis = intelligence.kpis
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [_heading(title, period, stamp), "", _headline(kpis)]

    drivers = _what_is_driving_risk(kpis, results)
    if drivers:
        sections += ["", drivers]

    hotspots = _repeat_exposures(intelligence.hotspots)
    if hotspots:
        sections += ["", hotspots]

    languages = _language_coverage(results)
    if languages:
        sections += ["", languages]

    attention = _needs_attention_now(results)
    sections += ["", attention, "", _DISCLAIMER]
    return "\n".join(sections)


# -- sections ----------------------------------------------------------------

def _heading(title: str, period: str, stamp: str) -> str:
    line = title if not period else f"{title} - {period}"
    return f"{line}\nGenerated {stamp}\n" + "=" * min(len(line) + 10, 72)


def _headline(kpis: dict) -> str:
    total = int(kpis.get("total", 0) or 0)
    flagged = int(kpis.get("sif_potential", 0) or 0)
    rate = float(kpis.get("sif_rate", 0.0) or 0.0)
    mean_risk = float(kpis.get("mean_risk", 0.0) or 0.0)
    critical = int(kpis.get("critical", 0) or 0)
    outstanding = int(kpis.get("needs_review", 0) or 0)

    if not total:
        return "No reports have been analysed for this period."

    lines = [
        f"{total} report(s) analysed. {flagged} carry serious injury or fatality "
        f"(SIF) potential ({rate:.1f}% of the corpus), mean risk {mean_risk:.1f}/100."
    ]
    if critical:
        lines.append(f"{critical} scored in the Critical band and required "
                     "verification before driving an intervention.")
    if outstanding:
        lines.append(f"{outstanding} report(s) are still awaiting an expert decision.")
    else:
        lines.append("Every report that needed a person has been decided.")
    return " ".join(lines)


def _what_is_driving_risk(kpis: dict, results: Sequence["PipelineResult"]) -> str:
    flagged = [item for item in results if item.sif_potential]
    if not flagged:
        return ""
    top_rule = str(kpis.get("top_rule") or "").strip()
    energies: dict = {}
    barriers: dict = {}
    for item in flagged:
        if item.energy_source:
            energies[item.energy_source] = energies.get(item.energy_source, 0) + 1
        if item.barrier_failure:
            barriers[item.barrier_failure] = barriers.get(item.barrier_failure, 0) + 1
    top_energy = max(energies, key=energies.get) if energies else ""
    top_barrier = max(barriers, key=barriers.get) if barriers else ""

    lines = ["WHAT IS DRIVING RISK"]
    if top_rule and top_rule != "-":
        lines.append(f"- Most-cited Life-Saving Rule: {top_rule}.")
    if top_energy:
        lines.append(f"- Most common energy source in flagged reports: {top_energy} "
                     f"({energies[top_energy]} of {len(flagged)}).")
    if top_barrier:
        lines.append(f"- Most common failed barrier: {top_barrier} "
                     f"({barriers[top_barrier]} of {len(flagged)}). Fix the control, "
                     "not the incident.")
    return "\n".join(lines)


def _repeat_exposures(hotspots: Sequence) -> str:
    if not hotspots:
        return ""
    lines = ["REPEAT EXPOSURES (ranked by SIF-precursor density, not volume)"]
    for spot in list(hotspots)[:_MAX_LISTED_ITEMS]:
        lines.append(f"- {spot.kind}: {spot.label} - {spot.sif_reports}/{spot.reports} "
                     f"report(s) carry SIF potential ({spot.sif_rate:.0f}% density).")
    remaining = len(hotspots) - _MAX_LISTED_ITEMS
    if remaining > 0:
        lines.append(f"... and {remaining} more on the Risk hotspots page.")
    return "\n".join(lines)


def _language_coverage(results: Sequence["PipelineResult"]) -> str:
    translated = [item for item in results if item.source_language]
    if not translated:
        return ""
    languages: dict = {}
    for item in translated:
        languages[item.source_language] = languages.get(item.source_language, 0) + 1
    breakdown = ", ".join(f"{language} ({count})"
                          for language, count in sorted(languages.items()))
    return (f"LANGUAGE COVERAGE\n- {len(translated)} report(s) were not written in "
            f"English and were translated for analysis: {breakdown}. The original "
            "wording is kept as the record for each.")


def _needs_attention_now(results: Sequence["PipelineResult"]) -> str:
    outstanding = [item for item in results if item.needs_review]
    if not outstanding:
        return "NEEDS ATTENTION NOW\n- Nothing outstanding. The review queue is clear."

    outstanding = sorted(outstanding, key=lambda item: -item.risk_score)
    lines = ["NEEDS ATTENTION NOW"]
    for item in outstanding[:_MAX_LISTED_ITEMS]:
        reference = item.reference or "unreferenced report"
        lines.append(f"- {reference} [{item.review_trigger}] risk {item.risk_score:.0f}/100: "
                     f"{item.review_reason}")
    remaining = len(outstanding) - _MAX_LISTED_ITEMS
    if remaining > 0:
        lines.append(f"... and {remaining} more in Human review.")
    return "\n".join(lines)
