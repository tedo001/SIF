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
from typing import TYPE_CHECKING, Mapping, Sequence

from .lexical import (NO_BARRIER_FAILURE, NO_ENERGY, UNKNOWN_ACTIVITY,
                      UNKNOWN_LOCATION)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import Intelligence, PipelineResult

__all__ = ["report_brief", "plain_brief", "corpus_bulletin"]

#: Every generated document carries this - the console's own description of
#: itself, not a claim about what the reader should do with the output.
_DISCLAIMER = "Prototype output - for review, not a statutory record."

#: How many outstanding items the bulletin names individually before it falls
#: back to a count. A bulletin that lists forty rows is not read.
_MAX_LISTED_ITEMS = 10


#: Placeholders the extractors use when a field could not be filled. They are
#: honest in a table column and noise in a sentence, so a brief drops them.
_UNSAID = {UNKNOWN_ACTIVITY.lower(), UNKNOWN_LOCATION.lower(),
           NO_BARRIER_FAILURE.lower(), NO_ENERGY.lower(), "-", ""}


def _lead_lower(text: str) -> str:
    """Lowercase only the first letter, so LOTO and PTW survive mid-sentence."""
    return text[:1].lower() + text[1:] if text else text


def _said(fields: Mapping[str, object], key: str) -> str:
    """One field, or '' when it holds a not-stated placeholder."""
    value = str(fields.get(key) or "").strip()
    return "" if value.lower() in _UNSAID else value


def plain_brief(fields: Mapping[str, object]) -> str:
    """A short brief on one report, in the words an HSE reader would use.

    Short sentences, no jargon the reader has to decode, and the consequence
    first: someone deciding whether a report could have killed a person should
    not have to infer that from a rule name and a number. What follows is the
    reasoning behind it, then whatever the report did not say is simply left
    out rather than printed as "not stated".

    Takes a mapping rather than a :class:`~sif.pipeline.PipelineResult` so the
    interface can hand a table row straight in; :func:`report_brief` adapts a
    result onto it, which keeps one wording for both callers.

    Templated, never model-written: every clause is a field the pipeline
    already extracted, so nothing in the brief can outrun the record.
    """
    reference = str(fields.get("reference") or "").strip() or "This report"
    rule = str(fields.get("iogp_rule") or "").strip()
    if not rule:
        return f"{reference}: no text was extracted - nothing to summarise."

    energy = _said(fields, "energy_source")
    barrier = _said(fields, "barrier_failure")
    try:
        risk = float(fields.get("risk_score") or 0.0)
    except (TypeError, ValueError):
        risk = 0.0
    band = str(fields.get("risk_band") or "Low").strip()

    lines = []
    if fields.get("sif_potential"):
        lines.append(f"{reference} carries fatal potential: someone could have been "
                     "killed or seriously hurt.")
        if energy and barrier:
            lines.append(f"{energy} was uncontrolled because {_lead_lower(barrier)}.")
        elif barrier:
            lines.append(f"A critical control failed: {_lead_lower(barrier)}.")
        elif energy:
            lines.append(f"{energy} was present and not adequately controlled.")
    else:
        lines.append(f"{reference}: no serious-injury or fatality precursor was found.")
        if fields.get("high_energy") and not fields.get("barrier_failed"):
            lines.append(f"{energy or 'A high-energy source'} was present, but no failed "
                         "control was recognised - either the work was controlled, or the "
                         "barrier is described in words the system has not learned yet.")

    lines.append(f"Rule: {rule}. Risk {risk:.0f} out of 100 ({band} band).")

    context = [part for part in (_said(fields, "activity"), _said(fields, "location")) if part]
    if context:
        lines.append("Seen during " + " at ".join(context) + "."
                     if len(context) == 2 else f"Seen during {context[0]}.")

    if fields.get("minimizing_language"):
        lines.append("The report plays this down in its own words. The finding rests on "
                     "the facts above, not on how it was written.")
    language = str(fields.get("source_language") or "").strip()
    if language:
        lines.append(f"Translated from {language} for analysis; the original wording "
                     "stays the record.")
    if fields.get("needs_review"):
        reason = str(fields.get("review_reason") or "").strip()
        lines.append(f"A person still has to check this{': ' + reason if reason else '.'}")
    return " ".join(lines)


def report_brief(result: "PipelineResult") -> str:
    """The same brief, for a :class:`~sif.pipeline.PipelineResult`."""
    return plain_brief(result.to_dict())


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
