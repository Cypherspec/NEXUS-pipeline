"""ADMET screening rules, evaluated on this package's own descriptors.

Each endpoint below is a published closed-form rule or regression rather than a
trained model, so every number a panel shows can be traced to its source and
checked by hand:

    aqueous solubility   ESOL form, Delaney (2004), J. Chem. Inf. Comput. Sci. 44,
                         1000, with coefficients refitted on ESOL by
                         benchmarks/fit_esol.py (see `esol_log_solubility`)
    passive absorption   Egan, Merz & Baldwin (2000), J. Med. Chem. 43, 3867
    CNS exposure         Pajouhesh & Lenz (2005), NeuroRx 2, 541
    hERG liability       Aronov (2005), Drug Discov. Today 10, 149
    in vivo tolerability  Hughes et al. (2008), Bioorg. Med. Chem. Lett. 18, 4872
    overall ADMET risk   Gleeson (2008), J. Med. Chem. 51, 817

Scope, stated plainly because the distinction matters when reading the panel:
these are *structure-derived screening flags*. Only the solubility estimate is
quantitative, and only it is checked against measured values (tests/test_admet.py
scores it on the ESOL set). The flags say a molecule sits inside or outside a
range where compounds have historically behaved well; they do not predict an
assay result for any individual compound, and they are not a safety assessment.
Metabolism by specific CYP isoforms, transporter interaction and excretion are
not modelled at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus.chem.admet_params import (ESOL_AROMATIC_PROPORTION, ESOL_INTERCEPT, ESOL_LOGP,
                                     ESOL_MW, ESOL_ROTATABLE_BONDS, ESOL_TEST_R2,
                                     ESOL_TEST_RMSE)
from nexus.chem.descriptors import (count_functional_groups, crippen_logp,
                                    num_lipinski_hbd, num_rotatable_bonds, tpsa)
from nexus.chem.mol import Molecule

# A risk label is one of these three, ordered. Panels colour on this, not on the
# underlying numbers, so the vocabulary stays the same across endpoints.
LOW, MODERATE, HIGH = "low", "moderate", "high"


def aromatic_proportion(mol: Molecule) -> float:
    """Fraction of heavy atoms that are aromatic, the ESOL `AP` term."""
    heavy = [a for a in mol.atoms if a.symbol != "H"]
    if not heavy:
        return 0.0
    return sum(1 for a in heavy if a.aromatic) / len(heavy)


def num_basic_amines(mol: Molecule) -> int:
    """Aliphatic amines, the centres that carry a positive charge at pH 7.4.

    Amides and anilines are excluded by `count_functional_groups`, which is the
    distinction that matters here: it is the protonated aliphatic nitrogen that
    is associated with hERG binding, not nitrogen in general.
    """
    groups = count_functional_groups(mol)
    return (groups["amine_primary"] + groups["amine_secondary"]
            + groups["amine_tertiary"])


def esol_log_solubility(mol: Molecule) -> float:
    """log10 of aqueous solubility in mol/L, in Delaney's four-descriptor form.

    The form is Delaney's; the coefficients are re-estimated on ESOL against
    this package's own logP by benchmarks/fit_esol.py, because the published
    coefficients assume Delaney's clogP and leave a +0.98 log-unit bias when
    used with ours.

    Held-out accuracy on a scaffold split is RMSE 1.30, R^2 0.63 -- better than
    the published coefficients on the same split (1.73, 0.34) and than
    predicting the mean (2.30), but well short of Delaney's own reported ~0.75
    RMSE. The gap is inherited from the logP input, which is itself an estimate
    (held-out R^2 0.21). Read a value as a bracket of roughly a log unit either
    way, and use `solubility_class` rather than the number when triaging.
    """
    return (ESOL_INTERCEPT
            + ESOL_LOGP * crippen_logp(mol)
            + ESOL_MW * mol.molecular_weight()
            + ESOL_ROTATABLE_BONDS * num_rotatable_bonds(mol)
            + ESOL_AROMATIC_PROPORTION * aromatic_proportion(mol))


def solubility_class(log_s: float) -> str:
    """Delaney's own bands over log S: poorly, moderately or highly soluble."""
    if log_s < -6.0:
        return "insoluble"
    if log_s < -4.0:
        return "poorly soluble"
    if log_s < -2.0:
        return "moderately soluble"
    return "soluble"


def egan_absorption(mol: Molecule) -> bool:
    """Inside Egan's "egg" for passive gut absorption: TPSA <= 131.6, logP <= 5.88."""
    return tpsa(mol) <= 131.6 and crippen_logp(mol) <= 5.88


def cns_exposure(mol: Molecule) -> tuple[bool, list[str]]:
    """Whether a molecule sits in the range where CNS drugs cluster.

    Pajouhesh & Lenz surveyed marketed CNS agents and found them concentrated in
    a narrow window. Falling outside it is common among compounds that do not
    reach the brain, which is why it is worth flagging, but the window is
    descriptive: plenty of peripherally acting drugs sit inside it too.

    Returns the verdict and the properties that fall outside the window.
    """
    failures = []
    if mol.molecular_weight() > 450.0:
        failures.append("MW > 450")
    if tpsa(mol) > 90.0:
        failures.append("TPSA > 90")
    if not 1.0 <= crippen_logp(mol) <= 4.0:
        failures.append("logP outside 1-4")
    if num_lipinski_hbd(mol) > 3:
        failures.append("HBD > 3")
    return (not failures), failures


# This package's logP is fitted against logD at pH 7.4 (MoleculeNet
# Lipophilicity), so for a basic amine it reads one to two units below the
# neutral logP that the published rules below are stated in: at pH 7.4 an amine
# of pKa ~9 is largely protonated, and Henderson-Hasselbalch puts
# logD = logP - log10(1 + 10^(pKa - 7.4)), about -1.6 at pKa 9. No pKa is
# computed here, so a single average offset is added back for molecules that
# carry a basic centre. It is a coarse correction on a coarse flag, and it is
# the reason `herg_liability` is banded rather than numeric.
AMINE_LOGD_TO_LOGP = 1.5


def neutral_logp(mol: Molecule) -> float:
    """Lipophilicity on the neutral-logP scale the published rules assume."""
    log_d = crippen_logp(mol)
    return log_d + AMINE_LOGD_TO_LOGP if num_basic_amines(mol) > 0 else log_d


def herg_liability(mol: Molecule) -> str:
    """Risk band for hERG channel binding, from basicity and lipophilicity.

    A basic centre plus high lipophilicity is the combination most consistently
    reported among hERG actives, and it is the pair Aronov's review singles out.
    The cut-offs are his, applied to `neutral_logp`.

    This is a triage flag on two descriptors. hERG binding also depends on the
    distance between the basic centre and the aromatic rings that pack into the
    channel, which is not modelled here, and the bands have not been calibrated
    against measured hERG data -- no such set ships with this package. A `low`
    is therefore weak evidence, and a `high` says only that the compound should
    meet a real assay before it is advanced.
    """
    log_p = neutral_logp(mol)
    basic = num_basic_amines(mol) > 0
    if basic and log_p >= 3.7:
        return HIGH
    if basic and log_p >= 2.5:
        return MODERATE
    if log_p >= 4.5:
        return MODERATE
    return LOW


def pfizer_3_75(mol: Molecule) -> str:
    """Hughes' 3/75 rule: logP > 3 with TPSA < 75 marks a riskier region.

    In Pfizer's retrospective, compounds in that quadrant were several times
    more likely to show toxicity in preclinical studies. It is an enrichment
    over a large set, not a property of any single compound.
    """
    if crippen_logp(mol) > 3.0 and tpsa(mol) < 75.0:
        return HIGH
    if crippen_logp(mol) > 3.0 or tpsa(mol) < 75.0:
        return MODERATE
    return LOW


def gleeson_4_400(mol: Molecule) -> bool:
    """Gleeson's 4/400: MW <= 400 and logP <= 4, where ADMET behaviour is better on average."""
    return mol.molecular_weight() <= 400.0 and crippen_logp(mol) <= 4.0


def oral_bioavailability_flags(mol: Molecule) -> list[str]:
    """Veber's two criteria for oral bioavailability, as the list that fails."""
    failures = []
    if num_rotatable_bonds(mol) > 10:
        failures.append("rotatable bonds > 10")
    if tpsa(mol) > 140.0:
        failures.append("TPSA > 140")
    return failures


@dataclass
class ADMETProfile:
    """Every endpoint for one molecule, with the inputs they were derived from."""

    log_solubility: float
    solubility_class: str
    absorption_egan: bool
    cns_candidate: bool
    cns_failures: list[str]
    herg_risk: str
    tox_risk_3_75: str
    gleeson_pass: bool
    bioavailability_failures: list[str]
    risk_flags: int

    def as_dict(self) -> dict:
        return {
            "logSolubility": round(self.log_solubility, 3),
            "solubilityClass": self.solubility_class,
            "absorptionEgan": self.absorption_egan,
            "cnsCandidate": self.cns_candidate,
            "cnsFailures": self.cns_failures,
            "hergRisk": self.herg_risk,
            "toxRisk375": self.tox_risk_3_75,
            "gleesonPass": self.gleeson_pass,
            "bioavailabilityFailures": self.bioavailability_failures,
            "riskFlags": self.risk_flags,
        }


def admet_profile(mol: Molecule) -> ADMETProfile:
    """Run every endpoint over one molecule.

    `risk_flags` counts how many of the independent screens the molecule fails,
    from 0 to 5. It is a sort key for triaging a list, deliberately a plain
    count: the screens disagree about what matters and weighting them would
    imply a calibration none of them carries.
    """
    log_s = esol_log_solubility(mol)
    cns_ok, cns_failures = cns_exposure(mol)
    herg = herg_liability(mol)
    tox = pfizer_3_75(mol)
    gleeson = gleeson_4_400(mol)
    bioavailability = oral_bioavailability_flags(mol)

    flags = 0
    flags += 1 if log_s < -4.0 else 0
    flags += 0 if egan_absorption(mol) else 1
    flags += 1 if herg == HIGH else 0
    flags += 1 if tox == HIGH else 0
    flags += 1 if bioavailability else 0

    return ADMETProfile(
        log_solubility=log_s,
        solubility_class=solubility_class(log_s),
        absorption_egan=egan_absorption(mol),
        cns_candidate=cns_ok,
        cns_failures=cns_failures,
        herg_risk=herg,
        tox_risk_3_75=tox,
        gleeson_pass=gleeson,
        bioavailability_failures=bioavailability,
        risk_flags=flags,
    )


def admet_summary(mols) -> dict:
    """Aggregate counts over a set, which is what a map-wide panel reports."""
    profiles = [admet_profile(m) for m in mols]
    if not profiles:
        return {"n": 0}
    n = len(profiles)
    return {
        "n": n,
        "meanLogSolubility": round(sum(p.log_solubility for p in profiles) / n, 3),
        "absorbedFraction": round(sum(p.absorption_egan for p in profiles) / n, 3),
        "cnsFraction": round(sum(p.cns_candidate for p in profiles) / n, 3),
        "hergHigh": sum(p.herg_risk == HIGH for p in profiles),
        "hergModerate": sum(p.herg_risk == MODERATE for p in profiles),
        "toxHigh": sum(p.tox_risk_3_75 == HIGH for p in profiles),
        "gleesonPassFraction": round(sum(p.gleeson_pass for p in profiles) / n, 3),
        "cleanFraction": round(sum(p.risk_flags == 0 for p in profiles) / n, 3),
    }
