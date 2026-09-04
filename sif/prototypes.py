"""Label prototypes for zero-shot semantic classification.

The heads classify without labelled training data by embedding a handful of
natural-language descriptions per label and comparing a report's sentences
against them. Prototypes are written the way an HSE professional would describe
the hazard in plain English - deliberately *not* as keyword lists, since the
whole point of the semantic layer is to catch phrasing the lexical patterns
miss.

Rule keys must match the rule names in :mod:`sif.lexical` so the lexical and
semantic scores can be fused label-for-label.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "RULE_PROTOTYPES",
    "ENERGY_PROTOTYPES",
    "BARRIER_PROTOTYPES",
    "ACTIVITY_PROTOTYPES",
    "LOCATION_PROTOTYPES",
]

RULE_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "Working at Height": (
        "A worker is elevated above ground on a scaffold, platform, ladder, roof or "
        "derrick and could fall to a lower level.",
        "Fall protection such as a full body harness, lanyard, anchor point or guardrail "
        "is involved in the work.",
    ),
    "Energy Isolation": (
        "Electrical or stored energy was not isolated, earthed or locked out before work "
        "began on the equipment.",
        "Work on a live cable, high voltage feeder, switchgear, breaker or motor control "
        "centre without proving it dead.",
    ),
    "Line of Fire": (
        "A person is positioned in the path of a moving, falling or released object, or "
        "under a suspended load.",
        "Stored pressure, a whipping hose or a dropped object could strike someone standing "
        "in the danger zone.",
    ),
    "Confined Space": (
        "Entry into a tank, vessel, sump, pit or other confined space with a restricted "
        "means of escape.",
        "The atmosphere inside the space may be oxygen deficient or contain toxic gas such "
        "as hydrogen sulphide, requiring gas testing and ventilation.",
    ),
    "Safe Mechanical Lifting": (
        "A crane, hydra, hoist or forklift is lifting a load with slings, shackles and "
        "rigging gear.",
        "A lifting operation with a load chart, lift plan, certified rigger and tag lines.",
    ),
    "Hot Work": (
        "Welding, gas cutting or grinding produces sparks and flame near hydrocarbons.",
        "An ignition source is introduced into an area that may contain flammable vapour, "
        "requiring a gas free certificate and a fire watch.",
    ),
    "Driving": (
        "A vehicle, tanker or truck is being driven on site or on a public road.",
        "Driving behaviour such as speeding, seat belt use, reversing with a banksman or "
        "journey management is at issue.",
    ),
    "Bypassing Safety Controls": (
        "A safety device such as an interlock, trip, alarm, emergency shutdown or machine "
        "guard was bypassed, inhibited, disabled or removed.",
        "Protective systems were defeated so that production or maintenance work could "
        "continue.",
    ),
    "Work Authorisation": (
        "Work started without a valid permit to work, job safety analysis or risk "
        "assessment.",
        "The permit had expired, was not signed, or the crew was never briefed on the "
        "hazards of the task.",
    ),
    "Excavation & Ground Disturbance": (
        "Digging, trenching or excavation work that could collapse on a person or strike a "
        "buried cable or pipeline.",
        "Ground disturbance without shoring, benching or an underground utility survey.",
    ),
    "Well Control & Process Containment": (
        "Loss of containment of hydrocarbons from a wellhead, flow line, separator, flange "
        "or pipeline.",
        "Well control equipment such as the blow out preventer, or a gas or crude oil leak "
        "from pressurised process equipment.",
    ),
}

ENERGY_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "Gravity / Fall from height": (
        "A person or object could fall from an elevated position to a lower level.",
        "Work several metres above ground on a platform, scaffold, roof or open grating.",
    ),
    "Electrical energy": (
        "High voltage electrical energy is present in a live conductor, panel or cable.",
        "Equipment is energised and could deliver an electric shock or arc flash.",
    ),
    "Suspended load / Mechanical": (
        "A heavy load is suspended from a crane hook or held by rigging gear.",
        "Rotating or moving mechanical equipment could crush, entangle or strike a person.",
    ),
    "Pressure / Stored energy": (
        "Pressurised fluid, gas or steam is contained and could be released violently.",
        "Stored energy remains in a line, vessel or hose that has not been depressurised.",
    ),
    "Fire / Explosion": (
        "Flammable hydrocarbon vapour is present together with an ignition source.",
        "Hot work sparks or naked flame near combustible material could start a fire.",
    ),
    "Toxic / Asphyxiant atmosphere": (
        "The atmosphere contains toxic gas such as hydrogen sulphide, or is oxygen "
        "deficient.",
        "Breathing the air in the area could cause poisoning or asphyxiation.",
    ),
    "Vehicle / Traffic motion": (
        "A moving vehicle could collide with a person, another vehicle or a structure.",
        "Heavy vehicle movement, reversing or speeding on site roads.",
    ),
    "Excavation / Ground collapse": (
        "The wall of a trench or excavation could collapse and bury a person.",
        "Unsupported soil around a deep excavation where people are working.",
    ),
    "Thermal energy": (
        "A hot surface, hot oil, steam or molten material could cause severe burns.",
        "Very high or very low temperature process fluid is exposed.",
    ),
}

BARRIER_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "Fall protection not used / not anchored": (
        "The worker had no harness, or the lanyard was not hooked to an anchor point.",
        "Guardrails or edge protection were missing, removed or incomplete.",
    ),
    "Energy isolation / LOTO not applied or verified": (
        "The equipment was not isolated, locked out, earthed or proved dead before work.",
        "The circuit was left energised or the isolation was never verified.",
    ),
    "Permit to work / JSA absent, expired or not followed": (
        "There was no valid permit to work, or the permit had expired.",
        "No job safety analysis, risk assessment or toolbox talk was carried out before "
        "starting.",
    ),
    "Gas testing / ventilation / atmospheric control missing": (
        "No gas test was carried out before or during entry.",
        "Ventilation, breathing apparatus or a standby hole watch was missing.",
    ),
    "Exclusion zone / barricading absent": (
        "The area was not barricaded and people could walk into the danger zone.",
        "Personnel stood under a suspended load or inside the swing radius with no banksman.",
    ),
    "Safety device bypassed, inhibited or removed": (
        "An interlock, trip, alarm or emergency shutdown was bypassed or inhibited.",
        "A machine guard was removed or left open while the equipment ran.",
    ),
    "Fire prevention controls not in place": (
        "No fire watch, extinguisher or fire blanket was provided for the hot work.",
        "Combustible material was not cleared before cutting or welding began.",
    ),
    "Mandatory PPE not worn": (
        "The worker was not wearing the required personal protective equipment.",
        "Helmet, gloves, goggles or safety footwear were not used for the task.",
    ),
    "Competence / supervision inadequate": (
        "The work was done by an untrained, uncertified or unauthorised person.",
        "No competent supervisor or certified operator was present.",
    ),
    "Equipment integrity / inspection lapse": (
        "The equipment was defective, damaged, frayed or corroded.",
        "Inspection, certification or calibration was overdue or expired.",
    ),
    "Traffic / journey management control breached": (
        "The driver was speeding or not wearing a seat belt.",
        "No journey management plan, banksman or authorised driver for the trip.",
    ),
    "Housekeeping / walkway integrity lapse": (
        "Spilled oil, water or material made the walkway slippery or obstructed.",
        "Poor housekeeping, loose grating or inadequate lighting on the access route.",
    ),
}

ACTIVITY_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "Confined space entry": ("Entering a tank, vessel or sump to work inside it.",),
    "Welding / grinding (hot work)": ("Welding, cutting or grinding metal.",),
    "Mechanical lifting operation": ("Lifting or shifting a load with a crane or hydra.",),
    "Excavation / ground disturbance": ("Digging a trench or excavating the ground.",),
    "Equipment maintenance": ("Servicing, overhauling or maintaining plant equipment.",),
    "Component replacement / repair": ("Replacing or repairing a damaged component.",),
    "Electrical maintenance work": ("Working on cables, panels or electrical equipment.",),
    "Scaffold erection / dismantling": ("Erecting or dismantling scaffolding.",),
    "Inspection / condition monitoring": ("Inspecting equipment or taking readings on rounds.",),
    "Vehicle movement / transport": ("Driving or transporting material on site.",),
    "Loading / unloading operation": ("Loading or unloading material or product.",),
    "Housekeeping / cleaning": ("Cleaning up an area or clearing spilled material.",),
    "Drilling / tripping operation": ("Drilling ahead or tripping pipe on the rig.",),
    "Line breaking / flange joint opening": ("Opening a flange or breaking into a line.",),
    "Construction / installation": ("Erecting, installing or fabricating new equipment.",),
}

LOCATION_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "Pump station": ("At a pump house or pumping station.",),
    "Group Gathering Station (GGS)": (
        "At a group gathering station where well fluid is collected.",),
    "Oil Collecting Station (OCS)": ("At an oil collecting station.",),
    "Well site / drilling location": ("At a well site, drill site or rig location.",),
    "Rig floor": ("On the rig floor or derrick floor.",),
    "Tank farm": ("In the tank farm among the storage tanks.",),
    "Electrical substation / MCC room": ("In the electrical substation, switchyard or MCC room.",),
    "Gas processing plant": ("In the gas compression or processing plant.",),
    "LPG plant": ("At the LPG bottling plant.",),
    "Workshop / fabrication yard": ("In the workshop or fabrication yard.",),
    "Pipeline right-of-way": ("Along the pipeline right of way in the field.",),
    "Effluent treatment plant": ("At the effluent treatment plant.",),
    "Camp / administrative area": ("In the camp, colony, canteen or office building.",),
    "Access / field road": ("On a field approach road or haul road.",),
    "Manifold area": ("At the manifold area of the process facility.",),
}
