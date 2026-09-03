"""Heuristic parsing engine for UA/UC and near-miss field reports.

Problem Statement 26165 (Oil India Limited) asks for a system that turns raw,
free-text Unsafe Act / Unsafe Condition (UA/UC) and near-miss reports into
structured, decision-grade safety intelligence.

The engine implemented here is deliberately dependency-free (standard library
only) and deterministic, so that it can be unit-tested, audited by an HSE
professional, and later swapped for / benchmarked against an ML classifier.

Analytical model
----------------
A report is flagged as **SIF-potential** (Serious Injury or Fatality potential)
only when both of the following are true, which mirrors the industry consensus
model used by IOGP / EI / Ops-Group programmes:

1. a **high-energy source** is present in the narrative (gravity, electrical,
   pressure, motion, mechanical, thermal, chemical, ...), and
2. a **critical barrier** was absent, bypassed, defeated or ineffective.

High energy without a barrier failure is normal, controlled work.  A barrier
failure without high energy is a housekeeping/quality issue.  The intersection
is where fatalities come from.

Public API
----------
    >>> from sif_engine import SIFEngine
    >>> SIFEngine().analyze("Worker on scaffold at 4 m without harness")["sif_potential"]
    True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

__all__ = [
    "SIFEngine",
    "SIFAssessment",
    "SEED_REPORTS",
    "CSV_TEXT_COLUMNS",
]

# ---------------------------------------------------------------------------
# Fallback constants -- every extraction routine degrades to one of these
# rather than raising, so a malformed report never breaks a batch run.
# ---------------------------------------------------------------------------

UNCLASSIFIED_RULE = "Unclassified / General HSE"
UNKNOWN_ACTIVITY = "Unspecified activity"
UNKNOWN_LOCATION = "Location not stated"
NO_BARRIER_FAILURE = "No barrier failure identified"
NO_ENERGY = "No high-energy source identified"

# Column names accepted by the batch CSV importer, in priority order.
CSV_TEXT_COLUMNS: Tuple[str, ...] = (
    "report",
    "description",
    "narrative",
    "text",
    "observation",
    "details",
    "incident",
)


def _compile(patterns: Sequence[str]) -> List[Pattern[str]]:
    """Compile a sequence of case-insensitive regex fragments."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleDefinition:
    """One IOGP Life-Saving Rule and the vocabulary that signals it."""

    name: str
    #: Strong signals -- unambiguous for this rule (weight 3).
    primary: Tuple[str, ...]
    #: Supporting signals -- shared with other rules (weight 1).
    secondary: Tuple[str, ...] = ()
    #: Energy category implied when this rule is matched.
    energy: str = ""


#: The nine IOGP Life-Saving Rules (2018 revision), plus the two Oil India
#: operational categories that recur in upstream UA/UC reporting.
IOGP_RULES: Tuple[RuleDefinition, ...] = (
    RuleDefinition(
        name="Working at Height",
        primary=(
            r"\bharness\b", r"\blanyard\b", r"\bscaffold\w*\b", r"\bfall arrest\b",
            r"\bfall protection\b", r"\banchor(?:age)? point\b", r"\bguard\s?rail\b",
            r"\bhand\s?rail\b", r"\bworking at height\b", r"\bwork(?:ing)? at heights?\b",
            r"\broof\b", r"\bderrick\b", r"\bmonkey board\b", r"\bmast\b",
            r"\bopen (?:floor )?grating\b", r"\bfloor opening\b", r"\bedge protection\b",
        ),
        secondary=(r"\bladder\b", r"\bplatform\b", r"\bfell\b", r"\bfall\b", r"\bclimb\w*\b",
                   r"\b\d+(?:\.\d+)?\s?(?:m|meters?|metres?|ft|feet)\b", r"\belevated\b"),
        energy="Gravity / Fall from height",
    ),
    RuleDefinition(
        name="Energy Isolation",
        primary=(
            r"\bloto\b", r"\block[\s-]?out\b", r"\btag[\s-]?out\b", r"\bisolat\w+\b",
            r"\bde[\s-]?energi[sz]\w*\b", r"\bearth(?:ing|ed)?\b", r"\bground(?:ing|ed)?\b",
            r"\bbreaker\b", r"\bmcc\b", r"\bswitch\s?gear\b", r"\bbus\s?bar\b",
            r"\blive (?:line|cable|circuit|conductor|panel)\b", r"\b(?:11|33|440|415|240)\s?kv?\b",
            r"\bhigh[\s-]?voltage\b", r"\bblind(?:ing|ed)?\b", r"\bspade\b",
            r"\bstored energy\b", r"\bline breaking\b", r"\bdepressuri[sz]\w*\b",
        ),
        secondary=(r"\bpanel\b", r"\bcable\b", r"\bmotor\b", r"\bpump\b", r"\bvalve\b",
                   r"\btransformer\b", r"\belectric\w*\b", r"\bshock\b"),
        energy="Electrical / Stored energy",
    ),
    RuleDefinition(
        name="Line of Fire",
        primary=(
            r"\bline of fire\b", r"\bsuspended load\b", r"\bunder the (?:load|hook|boom)\b",
            r"\bdropped object\b", r"\bstruck by\b", r"\bpinch point\b", r"\bsnap[\s-]?back\b",
            r"\bexclusion zone\b", r"\bbarricad\w+\b", r"\bpressuri[sz]ed line\b",
            r"\bstored pressure\b", r"\bhydro\s?test\b", r"\bblow[\s-]?out\b",
            r"\bwhipping hose\b", r"\bcrush(?:ed|ing)?\b",
        ),
        secondary=(r"\bhook\b", r"\bboom\b", r"\bswing\b", r"\btrajector\w+\b",
                   r"\bejected\b", r"\bpressure\b", r"\bpsi\b", r"\bbar\b", r"\bhose\b"),
        energy="Motion / Pressure release",
    ),
    RuleDefinition(
        name="Confined Space",
        primary=(
            r"\bconfined space\b", r"\bvessel entry\b", r"\btank entry\b", r"\bman\s?hole\b",
            r"\bgas test\w*\b", r"\bo2 (?:level|test)\b", r"\boxygen (?:level|deficien\w+)\b",
            r"\bh2s\b", r"\bventilat\w+\b", r"\bsba\b", r"\bscba\b", r"\bpurg(?:e|ed|ing)\b",
            r"\bhole watch\b", r"\bstandby ?man\b", r"\binert(?:ed|ing)?\b",
        ),
        secondary=(r"\btank\b", r"\bvessel\b", r"\bsump\b", r"\bpit\b", r"\bsilo\b",
                   r"\bsewer\b", r"\bexcavation\b", r"\batmosphere\b"),
        energy="Toxic / Oxygen-deficient atmosphere",
    ),
    RuleDefinition(
        name="Safe Mechanical Lifting",
        primary=(
            r"\bcrane\b", r"\brigging\b", r"\brigger\b", r"\bsling\b", r"\bshackle\b",
            r"\bhoist\w*\b", r"\bwinch\b", r"\bslew\w*\b", r"\btag ?line\b",
            r"\bswl\b", r"\bload chart\b", r"\bout\s?rigger\b", r"\blift plan\b",
            r"\bforklift\b", r"\bhydra\b",
        ),
        secondary=(r"\blift(?:ing)?\b", r"\bload\b", r"\bhook\b", r"\bboom\b", r"\bcasing\b"),
        energy="Suspended load / Mechanical",
    ),
    RuleDefinition(
        name="Hot Work",
        primary=(
            r"\bhot work\b", r"\bweld(?:ing|er)?\b", r"\bgrind(?:ing|er)\b", r"\bcutting torch\b",
            r"\bgas cutting\b", r"\bspark\w*\b", r"\bnaked flame\b", r"\bfire watch\b",
            r"\bflame proof\b", r"\bfire blanket\b", r"\bgas free\b", r"\bLEL\b",
        ),
        secondary=(r"\bflammable\b", r"\bhydrocarbon\b", r"\bfire extinguisher\b",
                   r"\bignition\b", r"\bvapou?r\b", r"\bcondensate\b"),
        energy="Fire / Ignition of hydrocarbons",
    ),
    RuleDefinition(
        name="Driving",
        primary=(
            r"\bseat\s?belt\b", r"\bover\s?speed\w*\b", r"\bspeed(?:ing| limit)\b",
            r"\bjourney management\b",
            r"\bIVMS\b",
            r"\bdriver\b",
            r"\bmobile phone while driving\b",
            r"\breversing\b", r"\bbanks?man\b", r"\bconvoy\b",
        ),
        secondary=(r"\bvehicle\b", r"\btruck\b", r"\btanker\b", r"\btrailer\b",
                   r"\bpick[\s-]?up\b", r"\bbus\b", r"\broad\b", r"\bcollision\b"),
        energy="Vehicle motion",
    ),
    RuleDefinition(
        name="Bypassing Safety Controls",
        primary=(
            r"\bbypass\w*\b", r"\boverrid\w+\b", r"\binhibit\w*\b", r"\bdefeat\w*\b",
            r"\bjumper(?:ed|ed out)?\b", r"\bforce[d]? (?:the )?(?:trip|alarm|interlock)\b",
            r"\binterlock\b", r"\besd\b", r"\bsafety valve\b", r"\bpsv\b", r"\brelief valve\b",
            r"\bgas detector (?:disabled|isolated|bypassed)\b", r"\btrip disabled\b",
            r"\bmachine guard\b", r"\bguard removed\b",
        ),
        secondary=(r"\balarm\b", r"\btrip\b", r"\bshut\s?down system\b", r"\bsensor\b"),
        energy="Process safety / Loss of containment",
    ),
    RuleDefinition(
        name="Work Authorisation",
        primary=(
            r"\bpermit to work\b", r"\bptw\b", r"\bwork permit\b", r"\bcold work permit\b",
            r"\btool\s?box talk\b", r"\btbt\b", r"\bjsa\b", r"\bjob safety analysis\b",
            r"\brisk assessment\b", r"\bmethod statement\b", r"\bexpired permit\b",
        ),
        secondary=(r"\bpermit\b", r"\bauthori[sz]\w+\b", r"\bsupervis\w+\b", r"\bbriefing\b"),
        energy="Uncontrolled work scope",
    ),
    RuleDefinition(
        name="Excavation & Ground Disturbance",
        primary=(
            r"\bexcavat\w+\b", r"\btrench\w*\b", r"\bshoring\b", r"\bbenching\b",
            r"\bcave[\s-]?in\b", r"\bburied (?:cable|pipeline|utility)\b",
            r"\bunderground (?:cable|pipeline|utility)\b", r"\bsoil collapse\b",
        ),
        secondary=(
            r"\bdig(?:ging)?\b",
            r"\bbackhoe\b",
            r"\bexcavator\b",
            r"\bpit\b",
            r"\bdepth\b",
        ),
        energy="Soil / Excavation collapse",
    ),
    RuleDefinition(
        name="Well Control & Process Containment",
        primary=(
            r"\bwell control\b", r"\bblow ?out preventer\b", r"\bbop\b", r"\bkick\b",
            r"\bgas leak\b", r"\bhydrocarbon (?:leak|release|spill)\b", r"\bloss of containment\b",
            r"\bwell head\b", r"\bchristmas tree\b", r"\bflow line\b", r"\bpigging\b",
            r"\bsour gas\b", r"\bcrude (?:oil )?(?:leak|spill)\b",
        ),
        secondary=(r"\bwell\b", r"\brig\b", r"\bworkover\b", r"\bseparator\b", r"\bmanifold\b",
                   r"\bflange\b", r"\bpipeline\b", r"\bgasket\b"),
        energy="Pressure / Hydrocarbon release",
    ),
)


@dataclass(frozen=True)
class EnergySignature:
    """A high-energy hazard category and the phrases that reveal it."""

    label: str
    patterns: Tuple[str, ...]


#: High-energy sources, aligned to the EI/IOGP "energy wheel".  Presence of any
#: of these is a necessary condition for SIF potential.
HIGH_ENERGY_SOURCES: Tuple[EnergySignature, ...] = (
    EnergySignature("Gravity / Fall from height", (
        r"\b(?:working )?at heights?\b", r"\bscaffold\w*\b", r"\bderrick\b", r"\bmast\b",
        r"\broof\b", r"\bladder\b", r"\bmonkey board\b", r"\bfloor opening\b",
        r"\bopen grating\b", r"\bfell from\b", r"\bfall(?:ing)? from\b", r"\belevated\b",
        r"\b(?:[3-9]|[1-9]\d)(?:\.\d+)?\s?(?:m|meters?|metres?)\b",
        r"\b(?:1[0-9]|[2-9]\d|\d{3})\s?(?:ft|feet)\b",
    )),
    EnergySignature("Electrical energy", (
        r"\bhigh[\s-]?voltage\b", r"\b(?:11|33|66|132|415|440|240)\s?kv?\b", r"\bkv\b",
        r"\blive (?:line|cable|circuit|conductor|panel|bus)\b", r"\bswitch\s?gear\b",
        r"\bbus\s?bar\b", r"\btransformer\b", r"\bmcc\b", r"\bbreaker\b", r"\benergi[sz]ed\b",
        r"\belectric(?:al)? shock\b", r"\barc flash\b",
    )),
    EnergySignature("Suspended load / Mechanical", (
        r"\bcrane\b", r"\bsuspended load\b", r"\bhoist\w*\b", r"\bwinch\b", r"\bsling\b",
        r"\bshackle\b", r"\bforklift\b", r"\bhydra\b", r"\brotating (?:equipment|shaft)\b",
        r"\bcoupling\b", r"\bflywheel\b", r"\bdrill (?:string|floor|pipe)\b", r"\bdraw\s?works\b",
        r"\bconveyor\b", r"\bunder the (?:load|hook|boom)\b", r"\bdropped object\b",
    )),
    EnergySignature("Pressure / Stored energy", (
        r"\bpressuri[sz]ed\b", r"\b\d{2,}\s?(?:psi|bar|kg/cm2)\b", r"\bhigh pressure\b",
        r"\bhydro\s?test\b", r"\bgas cylinder\b", r"\bnitrogen\b", r"\bsteam\b",
        r"\bwell ?head\b", r"\bbop\b", r"\bblow ?out\b", r"\bflow line\b", r"\bseparator\b",
        r"\bline breaking\b", r"\bstored energy\b", r"\bhose (?:burst|whip)\w*\b",
    )),
    EnergySignature("Fire / Explosion", (
        r"\bhot work\b", r"\bweld(?:ing|er)?\b", r"\bgrind(?:ing|er)\b", r"\bnaked flame\b",
        r"\bspark\w*\b", r"\bflammable\b", r"\bhydrocarbon vapou?r\b", r"\blel\b",
        r"\bgas leak\b", r"\bcondensate\b", r"\bignition source\b", r"\bfire\b",
    )),
    EnergySignature("Toxic / Asphyxiant atmosphere", (
        r"\bh2s\b", r"\bhydrogen sulphide\b", r"\bsour gas\b", r"\bconfined space\b",
        r"\boxygen deficien\w+\b", r"\bnitrogen purge\b", r"\btoxic\b", r"\bchlorine\b",
        r"\bammonia\b", r"\bfumes\b", r"\bcorrosive\b", r"\bacid\b", r"\bcaustic\b",
    )),
    EnergySignature("Vehicle / Traffic motion", (
        r"\bover\s?speed\w*\b", r"\bspeeding\b", r"\btanker\b", r"\btrailer\b",
        r"\breversing\b", r"\bcollision\b", r"\bhighway\b", r"\bheavy vehicle\b",
        r"\bdumper\b", r"\bcasing trailer\b",
    )),
    EnergySignature("Excavation / Ground collapse", (
        r"\bexcavat\w+\b", r"\btrench\w*\b", r"\bcave[\s-]?in\b", r"\bsoil collapse\b",
        r"\bburied (?:cable|pipeline|utility)\b",
    )),
    EnergySignature("Thermal energy", (
        r"\bhot (?:surface|oil|water|bitumen)\b", r"\bsteam (?:line|leak)\b", r"\bmolten\b",
        r"\bfurnace\b", r"\bheater treater\b", r"\bscald\w*\b", r"\b\d{3}\s?deg(?:rees)?\b",
    )),
)


@dataclass(frozen=True)
class BarrierSignature:
    """A critical barrier and the phrases that indicate it failed."""

    label: str
    patterns: Tuple[str, ...]


#: Explicit "barrier defeated" phrasing.  These are checked before the generic
#: negation sweep because they carry the barrier name with them.
CRITICAL_BARRIERS: Tuple[BarrierSignature, ...] = (
    BarrierSignature("Fall protection not used / not anchored", (
        r"\b(?:with)?out (?:a |the )?(?:full[\s-]?body )?harness\b",
        r"\bharness (?:was )?(?:not|un)(?:\s?worn|hooked|anchored|clipped|used)\b",
        r"\bno (?:fall (?:arrest|protection)|harness|lanyard|anchor(?:age)? point)\b",
        r"\blanyard (?:was )?not (?:hooked|anchored|clipped|attached)\b",
        r"\bunhooked (?:his|her|their|the) lanyard\b",
        r"\b(?:missing|no|without) (?:guard|hand)\s?rail\b",
        r"\b(?:guard|hand)\s?rail (?:was )?(?:missing|removed|damaged)\b",
        r"\bedge protection (?:missing|absent|removed)\b",
        r"\bincomplete scaffold\b", r"\bscaffold (?:tag|not) (?:missing|inspected|red)\b",
        r"\bunsecured (?:ladder|scaffold|platform)\b",
        r"\bfloor (?:opening|grating) (?:left )?open\b",
    )),
    BarrierSignature("Energy isolation / LOTO not applied or verified", (
        r"\b(?:with)?out (?:loto|lock\s?out|isolation|a permit)\b",
        r"\b(?:no|not) (?:locked|tagged) out\b",
        r"\bloto (?:not (?:applied|verified)|missing|bypassed|removed)\b",
        r"\b(?:not|never|un)\s?(?:de[\s-]?energi[sz]ed|isolated|earthed|grounded)\b",
        r"\bleft (?:un\s?grounded|ungrounded|un\s?earthed|energi[sz]ed|live)\b",
        r"\bstill (?:live|energi[sz]ed|pressuri[sz]ed|charged)\b",
        r"\bisolation (?:was )?(?:not verified|incomplete|missing|removed)\b",
        r"\bno (?:earthing|grounding|isolation certificate|caution board)\b",
        r"\bbreaker (?:not )?(?:racked out|left closed)\b",
        r"\btry[\s-]?out (?:test )?not (?:done|performed)\b",
    )),
    BarrierSignature("Permit to work / JSA absent, expired or not followed", (
        r"\b(?:with)?out (?:a |the |valid )?(?:permit|ptw|work permit|jsa|risk assessment)\b",
        r"\b(?:permit|ptw|jsa) (?:was )?"
        r"(?:not (?:raised|available|closed|signed)|expired|invalid|missing)\b",
        r"\bexpired permit\b",
        r"\bno (?:\w+\s+){0,3}(?:permit|ptw|jsa|toolbox talk|tbt|method statement)\b",
        r"\bno (?:\w+\s+){0,3}(?:survey|clearance|certificate|authori[sz]ation)"
        r" (?:had been |was )?(?:done|carried out|obtained|available)?\b",
        r"\btoolbox talk (?:was )?not (?:conducted|held|done)\b",
        r"\bwork (?:started|carried out) (?:with)?out (?:authori[sz]ation|approval)\b",
        r"\bnot (?:briefed|informed) (?:on|about)\b",
    )),
    BarrierSignature("Gas testing / ventilation / atmospheric control missing", (
        r"\bno gas (?:test|testing|detector|monitor)\b",
        r"\bgas test(?:ing)? (?:was )?not (?:done|carried out|performed|repeated)\b",
        r"\b(?:with)?out (?:gas test|forced ventilation|scba|breathing apparatus)\b",
        r"\bventilation (?:was )?(?:not|in)(?:adequate|stalled|provided| available)?\b",
        r"\bno (?:hole watch|standby ?man|attendant|rescue plan)\b",
        r"\bgas detector (?:bypassed|disabled|isolated|not calibrated)\b",
    )),
    BarrierSignature("Exclusion zone / barricading absent", (
        r"\b(?:no|without|missing) (?:barricad\w+|exclusion zone|cordon|hard "
        r"barrier|signage|warning sign|tape)\b",
        r"\barea (?:was )?not (?:barricaded|cordoned|isolated)\b",
        r"\bbarricad\w+ (?:was )?(?:removed|breached|inadequate|missing)\b",
        r"\bpersonnel (?:stood|standing|walked|working) (?:under|within)"
        r" (?:the )?(?:suspended )?(?:load|hook|boom|exclusion zone)\b",
        r"\bstood (?:directly )?under\b", r"\bunder the (?:suspended )?(?:load|hook|boom)\b",
        r"\bno banks?man\b", r"\bno flag ?man\b",
    )),
    BarrierSignature("Safety device bypassed, inhibited or removed", (
        r"\b(?:bypass\w*|overrid\w+|inhibit\w*|defeat\w*|jumper\w*)"
        r" (?:the )?(?:interlock|trip|alarm|esd|shutdown|sensor|detector|guard)\b",
        r"\b(?:interlock|trip|alarm|esd|shutdown|psv|relief valve|machine guard)\w*"
        r" (?:was )?(?:bypassed|inhibited|disabled|removed|defeated|isolated|not functional)\b",
        r"\bguard (?:was )?(?:removed|missing|open)\b",
        r"\bsafety (?:valve|device) (?:not|in)(?:stalled|operative| functional)?\b",
    )),
    BarrierSignature("Fire prevention controls not in place", (
        r"\bno fire (?:watch|extinguisher|blanket|water hose)\b",
        r"\bfire watch (?:was )?(?:absent|not (?:posted|available))\b",
        r"\b(?:with)?out (?:gas free certificate|hot work permit)\b",
        r"\bcombustibles? not (?:removed|cleared)\b",
        r"\bflammable material (?:nearby|not removed)\b",
    )),
    BarrierSignature("Mandatory PPE not worn", (
        r"\b(?:with)?out (?:ppe|helmet|hard hat|safety (?:shoes|goggles|glasses)|gloves|face "
        r"shield|ear plugs)\b",
        r"\bno (?:ppe|helmet|hard hat|safety (?:shoes|goggles|glasses)|gloves|face shield)\b",
        r"\b(?:helmet|hard hat|gloves|goggles|safety shoes) (?:was |were )?not (?:worn|used)\b",
        r"\bwearing (?:no|improper) ppe\b",
    )),
    BarrierSignature("Competence / supervision inadequate", (
        r"\b(?:un|not )(?:trained|certified|competent|qualified|authori[sz]ed) "
        r"(?:person|worker|operator|driver|rigger)\b",
        r"\bno (?:competent person|supervisor|rigger|certified operator)\b",
        r"\bsupervis\w+ (?:was )?(?:absent|not present|lacking)\b",
        r"\bcontractor (?:crew )?(?:not inducted|un\s?inducted)\b",
    )),
    BarrierSignature("Equipment integrity / inspection lapse", (
        r"\b(?:defective|damaged|frayed|corroded|worn[\s-]?out|cracked|leaking) "
        r"(?:sling|rope|hose|cable|scaffold|ladder|valve|flange|equipment|tool)\b",
        r"\b(?:inspection|certification|calibration|test) (?:was )?(?:overdue|expired|not "
        r"done|lapsed)\b",
        r"\bexpired (?:certificate|inspection|calibration|colour code)\b",
        r"\bun\s?inspected\b", r"\bno third[\s-]?party certificate\b",
    )),
    BarrierSignature("Traffic / journey management control breached", (
        r"\b(?:with)?out (?:a )?seat\s?belt\b", r"\bseat\s?belt not (?:worn|used|fastened)\b",
        r"\bexceed\w* (?:the )?speed limit\b", r"\bover\s?speed\w*\b",
        r"\bno journey (?:plan|management)\b", r"\bunauthori[sz]ed (?:driver|vehicle)\b",
        r"\breversing without (?:a )?(?:banks?man|guide)\b",
    )),
    BarrierSignature("Housekeeping / walkway integrity lapse", (
        r"\b(?:oil|water|mud|grease|chemical) spill(?:age)?\b", r"\bslippery\b",
        r"\bpoor housekeeping\b", r"\bobstruct\w+ (?:walkway|access|escape route)\b",
        r"\bmaterial (?:was )?(?:strewn|stacked|left)"
        r" (?:on|across) (?:the )?(?:walkway|access|path)\b",
        r"\binadequate (?:lighting|illumination)\b", r"\bloose (?:grating|chequered plate)\b",
    )),
)


#: Activity vocabulary -- longest / most specific phrases first.
ACTIVITY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bconfined space entry\b|\b(?:tank|vessel) entry\b",
     "Confined space entry"),
    (r"\bhot tapping\b",
     "Hot tapping"),
    (r"\bline breaking\b|\bbreaking (?:the )?(?:flange|joint|line)\b",
     "Line breaking / flange joint opening"),
    (r"\bpigging\b|\bpig (?:launch|receiv)\w*\b",
     "Pipeline pigging operation"),
    (r"\bwork\s?over\b|\bwell servicing\b",
     "Well workover operation"),
    (r"\bwire\s?line\b|\bslick\s?line\b|\bcoiled tubing\b",
     "Wireline / coiled tubing operation"),
    (r"\bdrilling\b|\btripping (?:in|out)\b|\bmaking up (?:the )?(?:stand|connection)\b",
     "Drilling / tripping operation"),
    (r"\bwell (?:head )?maintenance\b|\bchristmas tree\b",
     "Wellhead maintenance"),
    (r"\bscaffold(?:ing)? (?:erection|dismantl\w+)\b",
     "Scaffold erection / dismantling"),
    (r"\bpaint\w*\b|\bsand\s?blast\w*\b",
     "Painting / surface preparation"),
    (r"\bweld(?:ing|er)?\b|\bgas cutting\b|\bcutting torch\b|\bgrind(?:ing|er)\b",
     "Welding / grinding (hot work)"),
    (r"\blathe\b|\bmachining\b|\bturning a\b|\bmilling machine\b|\bbench grinder\b",
     "Machining / workshop operation"),
    (r"\bcrane (?:lift|operation)\b|\blifting operation\b|\brigging\b|\bhoisting\b",
     "Mechanical lifting operation"),
    (r"\bshifting (?:of )?(?:load|material|casing)\b",
     "Mechanical lifting operation"),
    (r"\bexcavat\w+\b|\btrench\w*\b|\bdigging\b",
     "Excavation / ground disturbance"),
    (r"\bpreventive maintenance\b|\bmaintenance (?:work|job|activity)\b|\bservicing\b",
     "Equipment maintenance"),
    (r"\boverhaul\w*\b",
     "Equipment maintenance"),
    (r"\breplac\w+\b|\brepair\w*\b|\bchang(?:e|ing) (?:the )?(?:fitting|bulb|lamp|gasket)\b",
     "Component replacement / repair"),
    (r"\bcable (?:laying|termination|jointing)\b|\belectrical (?:maintenance|work)\b",
     "Electrical maintenance work"),
    (r"\bmegger\w*\b",
     "Electrical maintenance work"),
    (r"\bcommissioning\b|\bstart[\s-]?up\b|\bshut\s?down job\b",
     "Commissioning / start-up"),
    (r"\btank (?:cleaning|gauging|dipping)\b",
     "Tank cleaning / gauging"),
    (r"\bloading\b|\bunloading\b|\bdecanting\b|\btanker filling\b",
     "Loading / unloading operation"),
    (r"\btransport\w*\b|\bdriving\b|\bjourney\b|\bconvoy\b",
     "Vehicle movement / transport"),
    (r"\binspection\b|\bsurvey\b|\bpatrolling\b|\brounds?\b|\bthermograph\w+\b",
     "Inspection / condition monitoring"),
    (r"\btaking readings?\b",
     "Inspection / condition monitoring"),
    (r"\bhousekeeping\b|\bcleaning\b",
     "Housekeeping / cleaning"),
    (r"\bwalking\b|\bwalkway\b|\baccess(?:ing)? (?:the )?(?:platform|area)\b",
     "Movement on site / access-egress"),
    (r"\bchemical (?:handling|dosing|injection)\b",
     "Chemical handling / dosing"),
    (r"\berection\b|\binstallation\b|\bfabrication\b|\bconstruction\b",
     "Construction / installation"),
)

#: Generic "while <verb>ing ..." capture used when no known activity matches.
_GERUND_RE = re.compile(
    r"\b(?:while|whilst|during|when)\s+(?:the\s+)?(?:\w+\s+){0,3}?(\w+ing\b(?:\s+\w+){0,3})",
    re.IGNORECASE,
)

#: Secondary fallback: an infinitive purpose clause, e.g. "to replace a light fitting".
_INFINITIVE_RE = re.compile(
    r"\bto\s+(?!the\b|a\b|an\b|his\b|her\b|their\b|its\b|be\b|it\b|him\b|them\b)"
    r"(\w+(?:\s+\w+){0,3})",
    re.IGNORECASE,
)

#: Site vocabulary typical of an upstream E&P asset (Oil India operations).
LOCATION_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    (r"\bgroup gathering station\b|\bggs[\s-]?\d*\b", "Group Gathering Station (GGS)"),
    (r"\boil collecting station\b|\bocs[\s-]?\d*\b", "Oil Collecting Station (OCS)"),
    (r"\bearly production system\b|\beps\b", "Early Production System (EPS)"),
    (r"\bpump(?:ing)? station\b|\bpump house\b", "Pump station"),
    (r"\bcompressor (?:station|house|shed)\b", "Compressor station"),
    (r"\bwell (?:site|pad)\b|\bwell no\.?\s?\w+\b|\brig site\b|\bdrill site\b",
     "Well site / drilling location"),
    (r"\brig floor\b|\bderrick floor\b|\bmud pit area\b|\bshale shaker\b", "Rig floor"),
    (r"\btank farm\b|\bstorage tank\b|\bcrude tank\b", "Tank farm"),
    (r"\bsub[\s-]?station\b|\bswitch\s?yard\b|\bmcc room\b|\bcontrol room\b",
     "Electrical substation / MCC room"),
    (r"\bLPG (?:plant|bottling)\b|\bbottling plant\b", "LPG plant"),
    (r"\bgas (?:compressor|processing|sweetening) plant\b|\bgas plant\b", "Gas processing plant"),
    (r"\brefinery\b|\bprocess plant\b", "Refinery / process plant"),
    (r"\bworkshop\b|\bfabrication yard\b|\bgarage\b", "Workshop / fabrication yard"),
    (r"\bwarehouse\b|\bstore ?yard\b|\bmaterial yard\b", "Warehouse / stores yard"),
    (r"\bpipeline (?:right of way|row|corridor|section)\b|\bpipeline\b", "Pipeline right-of-way"),
    (r"\bjetty\b|\bwharf\b|\bterminal\b", "Terminal / jetty"),
    (r"\bmanifold (?:area|station)\b|\bmanifold\b", "Manifold area"),
    (r"\beffluent (?:treatment )?plant\b|\betp\b", "Effluent treatment plant"),
    (r"\bcamp\b|\bcolony\b|\bcanteen\b|\boffice building\b", "Camp / administrative area"),
    (r"\bapproach road\b|\bfield road\b|\bhighway\b|\bhaul road\b|\bin[\s-]?field route\b",
     "Access / field road"),
    (r"\bduliajan\b", "Duliajan field HQ"),
    (r"\bdigboi\b", "Digboi asset"),
    (r"\bmoran\b", "Moran asset"),
    (r"\bkumchai\b", "Kumchai asset"),
    (r"\bjodhpur\b|\bjaisalmer\b", "Rajasthan asset"),
)

#: Explicit prepositional-phrase capture, e.g. "at the No. 3 pump station".
_LOCATION_PREP_RE = re.compile(
    r"\b(?:at|in|inside|near|on|adjacent to)\s+(?:the\s+)?"
    r"((?:[A-Z][\w\-/&.]*|no\.?|\d+)(?:\s+(?:[A-Za-z][\w\-/&.]*|\d+)){0,3})",
)

#: Severity vocabulary that reinforces (but never alone creates) SIF potential.
_SEVERITY_AMPLIFIERS = _compile((
    r"\bnear[\s-]?miss\b", r"\bcould have\b", r"\bnarrowly\b", r"\bjust missed\b",
    r"\bmissed (?:him|her|them|the worker) by\b", r"\bpotential(?:ly)? fatal\b",
    r"\bserious injury\b", r"\bfatalit\w+\b", r"\bhigh potential\b", r"\bhipo\b",
))

#: Vocabulary that marks a genuinely low-consequence observation.
_LOW_SEVERITY_MARKERS = _compile((
    r"\bminor\b", r"\bfirst[\s-]?aid\b", r"\bhousekeeping\b", r"\bcosmetic\b",
    r"\bno injury\b", r"\bslight\b", r"\bsuperficial\b",
))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SIFAssessment:
    """Structured output of :meth:`SIFEngine.analyze`.

    The five fields required by Problem Statement 26165 are ``sif_potential``,
    ``iogp_rule``, ``activity``, ``location`` and ``barrier_failure``.  The
    remaining fields are supporting evidence used by the dashboard and by any
    downstream audit of the engine's reasoning.
    """

    sif_potential: bool
    iogp_rule: str
    activity: str
    location: str
    barrier_failure: str
    energy_source: str = NO_ENERGY
    high_energy: bool = False
    barrier_failed: bool = False
    confidence: float = 0.0
    severity_hint: str = "Low"
    evidence: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return the assessment as a plain, JSON-serialisable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SIFEngine:
    """Rule-based parser that converts a raw field report into safety insight.

    The engine is stateless and thread-safe once constructed: all compiled
    patterns are built in ``__init__`` and only read afterwards, which lets a
    single instance be shared between the GUI thread and worker threads.

    Example
    -------
    >>> engine = SIFEngine()
    >>> result = engine.analyze(
    ...     "Near miss at GGS-4: 11 kV feeder was not earthed before the "
    ...     "electrician started cable jointing; no LOTO applied."
    ... )
    >>> result["iogp_rule"]
    'Energy Isolation'
    >>> result["sif_potential"]
    True
    """

    #: Minimum weighted score before a rule match is trusted.
    RULE_SCORE_THRESHOLD = 2

    def __init__(self) -> None:
        # Pre-compile every pattern group once; ``analyze`` then does pure reads.
        self._rules: List[Tuple[RuleDefinition, List[Pattern[str]], List[Pattern[str]]]] = [
            (rule, _compile(rule.primary), _compile(rule.secondary)) for rule in IOGP_RULES
        ]
        self._energies: List[Tuple[EnergySignature, List[Pattern[str]]]] = [
            (sig, _compile(sig.patterns)) for sig in HIGH_ENERGY_SOURCES
        ]
        self._barriers: List[Tuple[BarrierSignature, List[Pattern[str]]]] = [
            (sig, _compile(sig.patterns)) for sig in CRITICAL_BARRIERS
        ]
        self._activities: List[Tuple[Pattern[str], str]] = [
            (re.compile(pattern, re.IGNORECASE), label) for pattern, label in ACTIVITY_PATTERNS
        ]
        self._locations: List[Tuple[Pattern[str], str]] = [
            (re.compile(pattern, re.IGNORECASE), label) for pattern, label in LOCATION_KEYWORDS
        ]

    # -- public API --------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, object]:
        """Analyse one free-text report and return a structured dictionary.

        Parameters
        ----------
        text:
            Raw report narrative.  ``None``, empty or whitespace-only input is
            tolerated and yields a fully populated fallback result.

        Returns
        -------
        dict
            Keys: ``sif_potential`` (bool), ``iogp_rule``, ``activity``,
            ``location``, ``barrier_failure`` plus supporting evidence fields
            (see :class:`SIFAssessment`).
        """
        return self.assess(text).to_dict()

    def assess(self, text: str) -> SIFAssessment:
        """Same as :meth:`analyze` but returns the :class:`SIFAssessment` object."""
        clean = self._normalise(text)
        if not clean:
            return SIFAssessment(
                sif_potential=False,
                iogp_rule=UNCLASSIFIED_RULE,
                activity=UNKNOWN_ACTIVITY,
                location=UNKNOWN_LOCATION,
                barrier_failure=NO_BARRIER_FAILURE,
                raw_text="",
            )

        evidence: List[str] = []
        rule, rule_score = self._match_rule(clean, evidence)
        energy_label, energy_hits = self._match_energy(clean, evidence)
        barrier_label, barrier_hits = self._match_barrier(clean, evidence)

        # A rule whose own semantics imply high energy (e.g. Working at Height)
        # counts as an energy source even when the narrative is terse.
        high_energy = bool(energy_hits)
        if not high_energy and rule is not None and rule.energy and rule_score >= 3:
            high_energy = True
            energy_label = rule.energy
            evidence.append(f"energy inferred from rule '{rule.name}'")

        barrier_failed = bool(barrier_hits)
        sif_potential = high_energy and barrier_failed

        severity = self._severity_hint(clean, sif_potential)
        confidence = self._confidence(rule_score, energy_hits, barrier_hits)

        return SIFAssessment(
            sif_potential=sif_potential,
            iogp_rule=rule.name if rule is not None else UNCLASSIFIED_RULE,
            activity=self._extract_activity(clean),
            location=self._extract_location(text, clean),
            barrier_failure=barrier_label,
            energy_source=energy_label,
            high_energy=high_energy,
            barrier_failed=barrier_failed,
            confidence=confidence,
            severity_hint=severity,
            evidence=evidence,
            raw_text=text.strip() if isinstance(text, str) else "",
        )

    def analyze_many(self, texts: Sequence[str]) -> List[Dict[str, object]]:
        """Analyse a sequence of reports, skipping blank entries."""
        return [self.analyze(item) for item in texts if isinstance(item, str) and item.strip()]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalise(text: Optional[str]) -> str:
        """Lower-case, strip control characters and collapse whitespace."""
        if not isinstance(text, str):
            return ""
        collapsed = re.sub(r"\s+", " ", text).strip().lower()
        # Normalise common shorthand so a single pattern can catch both forms.
        collapsed = collapsed.replace("l.o.t.o", "loto").replace("p.t.w", "ptw")
        return collapsed

    def _match_rule(self, clean: str, evidence: List[str]) -> Tuple[Optional[RuleDefinition], int]:
        """Score every IOGP rule and return the best match above threshold."""
        best_rule: Optional[RuleDefinition] = None
        best_score = 0
        best_terms: List[str] = []

        for rule, primary, secondary in self._rules:
            terms = [m.group(0) for pattern in primary for m in [pattern.search(clean)] if m]
            score = 3 * len(terms)
            soft = [m.group(0) for pattern in secondary for m in [pattern.search(clean)] if m]
            score += len(soft)
            if score > best_score:
                best_rule, best_score, best_terms = rule, score, terms or soft

        if best_rule is None or best_score < self.RULE_SCORE_THRESHOLD:
            return None, best_score

        if best_terms:
            evidence.append("rule cues: " + ", ".join(sorted(set(best_terms))[:4]))
        return best_rule, best_score

    def _match_energy(self, clean: str, evidence: List[str]) -> Tuple[str, List[str]]:
        """Detect high-energy sources; returns (label, matched phrases)."""
        labels: List[str] = []
        hits: List[str] = []
        for signature, patterns in self._energies:
            matched = [m.group(0) for pattern in patterns for m in [pattern.search(clean)] if m]
            if matched:
                labels.append(signature.label)
                hits.extend(matched)
        if not hits:
            return NO_ENERGY, []
        evidence.append("energy cues: " + ", ".join(sorted(set(hits))[:4]))
        return " + ".join(dict.fromkeys(labels)), hits

    def _match_barrier(self, clean: str, evidence: List[str]) -> Tuple[str, List[str]]:
        """Detect failed critical barriers; returns (label, matched phrases)."""
        labels: List[str] = []
        hits: List[str] = []
        for signature, patterns in self._barriers:
            matched = [m.group(0) for pattern in patterns for m in [pattern.search(clean)] if m]
            if matched:
                labels.append(signature.label)
                hits.extend(matched)
        if not hits:
            return NO_BARRIER_FAILURE, []
        evidence.append("barrier cues: " + ", ".join(sorted(set(hits))[:4]))
        return "; ".join(dict.fromkeys(labels)), hits

    def _extract_activity(self, clean: str) -> str:
        """Map the narrative to a known task, else capture a '<verb>ing' phrase."""
        for pattern, label in self._activities:
            if pattern.search(clean):
                return label

        for fallback in (_GERUND_RE, _INFINITIVE_RE):
            match = fallback.search(clean)
            if match:
                phrase = match.group(1).strip(" ,.;")
                if phrase:
                    return phrase[:60].capitalize()
        return UNKNOWN_ACTIVITY

    def _extract_location(self, original: str, clean: str) -> str:
        """Resolve the site zone from vocabulary, else from a prepositional phrase."""
        for pattern, label in self._locations:
            if pattern.search(clean):
                return label

        if isinstance(original, str):
            prep = _LOCATION_PREP_RE.search(original)
            if prep:
                candidate = prep.group(1).strip(" ,.;")
                # Reject captures that are plainly not places.
                if candidate and not re.fullmatch(r"(?i)(the|a|an|he|she|they|it|no)", candidate):
                    return candidate[:60]
        return UNKNOWN_LOCATION

    @staticmethod
    def _severity_hint(clean: str, sif_potential: bool) -> str:
        """Coarse triage label used for dashboard colour-coding."""
        if any(pattern.search(clean) for pattern in _SEVERITY_AMPLIFIERS):
            return "High" if sif_potential else "Medium"
        if sif_potential:
            return "High"
        if any(pattern.search(clean) for pattern in _LOW_SEVERITY_MARKERS):
            return "Low"
        return "Medium"

    @staticmethod
    def _confidence(
        rule_score: int, energy_hits: Sequence[str], barrier_hits: Sequence[str]
    ) -> float:
        """Blend the three evidence streams into a 0.0-1.0 confidence value."""
        score = min(rule_score, 9) / 9.0 * 0.4
        score += min(len(energy_hits), 3) / 3.0 * 0.3
        score += min(len(barrier_hits), 3) / 3.0 * 0.3
        return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Seed data -- five realistic upstream E&P narratives for demo / presentation.
# ---------------------------------------------------------------------------

SEED_REPORTS: Tuple[str, ...] = (
    (
        "Near miss at Pump Station No. 3, Duliajan: during preventive maintenance of the "
        "booster pump the 11 kV feeder cable was left ungrounded and the breaker was not "
        "racked out. The electrician started cable jointing without LOTO applied and no "
        "isolation certificate was raised."
    ),
    (
        "Unsafe act observed at GGS-4 tank farm: a contract worker was standing on the "
        "incomplete scaffold at about 6 metres near the crude tank roof to replace a light "
        "fitting. He was wearing a full-body harness but the lanyard was not anchored and "
        "the guardrail on the working platform was missing."
    ),
    (
        "Unsafe condition during casing shifting at the drill site: the rigging crew and a "
        "helper stood directly under the suspended load while the hydra slewed the casing "
        "bundle. The area was not barricaded, there was no banksman, and one sling was "
        "found frayed with an expired third-party certificate."
    ),
    (
        "Unsafe condition reported in the OCS process area: minor oil spillage on the "
        "walkway near the manifold made the chequered plate slippery during routine rounds. "
        "Poor housekeeping, no injury; area was cleaned and anti-skid tape applied the same "
        "shift."
    ),
    (
        "Near miss during vessel entry at the Effluent Treatment Plant: two workers entered "
        "the separator sump for cleaning without gas testing being carried out and without "
        "forced ventilation. H2S was later detected at the manway and there was no hole "
        "watch posted; the permit had expired the previous shift."
    ),
)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    engine = SIFEngine()
    for index, report in enumerate(SEED_REPORTS, start=1):
        outcome = engine.analyze(report)
        print(f"[{index}] SIF={outcome['sif_potential']} | {outcome['iogp_rule']}")
        print(f"    activity : {outcome['activity']}")
        print(f"    location : {outcome['location']}")
        print(f"    barrier  : {outcome['barrier_failure']}")
        print(f"    energy   : {outcome['energy_source']} (conf {outcome['confidence']})")
