"""Free energy estimators checked against exactly solvable systems."""

import numpy as np
import pytest

from nexus.models.ddpm import (
    ATOM_VOCAB,
    BOND_VOCAB,
    DiscreteDiffusion,
    MoleculeDDPM,
    cosine_alpha_schedule,
    graph_to_molecule,
    molecule_to_graph,
)
from nexus.chem.smiles import parse_smiles, write_smiles
from nexus.physics.jarzynski import (
    SwitchingProtocol,
    bar_free_energy,
    crooks_free_energy,
    equilibrium_sample,
    jarzynski_bootstrap_ci,
    jarzynski_cumulant_estimate,
    jarzynski_free_energy,
    logsumexp,
    work_distribution_overlap,
)

# Harmonic oscillator whose spring constant changes from k0 to k1.
# F(k) = -(1/2beta) ln(2 pi / (beta k)), so dF = (1/2beta) ln(k1/k0) exactly.
K0, K1, BETA = 1.0, 4.0, 1.0
EXACT_DF = 0.5 * np.log(K1 / K0) / BETA


def harmonic_energy(x, lam):
    return 0.5 * ((1 - lam) * K0 + lam * K1) * np.sum(x ** 2, axis=1)


@pytest.fixture(scope="module")
def work_values():
    protocol = SwitchingProtocol(energy=harmonic_energy, beta=BETA,
                                 n_steps=200, dt=0.005)
    x0 = equilibrium_sample(harmonic_energy, 0.0, 3000, 1, BETA, seed=0)
    forward, _ = protocol.run(x0, forward=True, seed=1)
    x1 = equilibrium_sample(harmonic_energy, 1.0, 3000, 1, BETA, seed=2)
    reverse, _ = protocol.run(x1, forward=False, seed=3)
    return forward, reverse


def test_logsumexp_matches_direct_computation():
    v = np.array([1.0, 2.0, 3.0])
    assert logsumexp(v) == pytest.approx(np.log(np.sum(np.exp(v))))
    big = np.array([1000.0, 1000.0])
    assert np.isfinite(logsumexp(big))             # no overflow
    assert logsumexp(big) == pytest.approx(1000.0 + np.log(2))


def test_second_law_holds(work_values):
    forward, reverse = work_values
    assert forward.mean() >= EXACT_DF - 1e-9
    assert reverse.mean() >= -EXACT_DF - 1e-9


def test_jarzynski_recovers_exact_free_energy(work_values):
    forward, _ = work_values
    assert jarzynski_free_energy(forward, BETA) == pytest.approx(EXACT_DF, abs=0.05)


def test_jarzynski_is_never_above_the_mean_work(work_values):
    forward, _ = work_values
    assert jarzynski_free_energy(forward, BETA) <= forward.mean() + 1e-9


def test_bar_recovers_exact_free_energy(work_values):
    forward, reverse = work_values
    assert bar_free_energy(forward, reverse, BETA) == pytest.approx(EXACT_DF, abs=0.05)


def test_bar_is_at_least_as_accurate_as_jarzynski(work_values):
    forward, reverse = work_values
    err_bar = abs(bar_free_energy(forward, reverse, BETA) - EXACT_DF)
    err_jar = abs(jarzynski_free_energy(forward, BETA) - EXACT_DF)
    assert err_bar <= err_jar + 0.01


def test_crooks_crossing_recovers_free_energy(work_values):
    forward, reverse = work_values
    assert crooks_free_energy(forward, reverse, BETA) == pytest.approx(EXACT_DF, abs=0.08)


def test_cumulant_estimate_is_biased_for_nongaussian_work(work_values):
    forward, _ = work_values
    exponential = abs(jarzynski_free_energy(forward, BETA) - EXACT_DF)
    cumulant = abs(jarzynski_cumulant_estimate(forward, BETA) - EXACT_DF)
    assert cumulant > exponential


def test_bootstrap_interval_covers_the_exact_value(work_values):
    forward, _ = work_values
    low, high = jarzynski_bootstrap_ci(forward, BETA, n_boot=400, seed=0)
    assert low <= EXACT_DF <= high
    assert low < high


def test_work_overlap_is_a_fraction(work_values):
    forward, reverse = work_values
    overlap = work_distribution_overlap(forward, reverse)
    assert 0.0 <= overlap <= 1.0
    assert overlap > 0.3                    # the protocol is slow enough to overlap


def test_slower_switching_dissipates_less():
    x0 = equilibrium_sample(harmonic_energy, 0.0, 800, 1, BETA, seed=0)
    dissipation = []
    for n_steps in (40, 160, 640):
        protocol = SwitchingProtocol(energy=harmonic_energy, beta=BETA,
                                     n_steps=n_steps, dt=0.005)
        work, _ = protocol.run(x0, forward=True, seed=1)
        dissipation.append(work.mean() - EXACT_DF)
    assert dissipation[0] > dissipation[1] > dissipation[2] > -1e-9


def test_zero_free_energy_difference_for_a_shifted_well():
    """Translating a harmonic well leaves the partition function unchanged."""
    def shifted(x, lam):
        return 0.5 * np.sum((x - 2.0 * lam) ** 2, axis=1)

    protocol = SwitchingProtocol(energy=shifted, beta=1.0, n_steps=300, dt=0.005)
    x0 = equilibrium_sample(shifted, 0.0, 2000, 1, 1.0, seed=5)
    forward, _ = protocol.run(x0, forward=True, seed=6)
    x1 = equilibrium_sample(shifted, 1.0, 2000, 1, 1.0, seed=7)
    reverse, _ = protocol.run(x1, forward=False, seed=8)
    assert bar_free_energy(forward, reverse, 1.0) == pytest.approx(0.0, abs=0.08)


def test_empty_work_arrays_return_nan():
    assert np.isnan(jarzynski_free_energy(np.array([])))
    assert np.isnan(bar_free_energy(np.array([]), np.array([1.0])))


# ------------------------------------------------------------------- diffusion

def test_cosine_schedule_is_monotone_and_bounded():
    alpha = cosine_alpha_schedule(100)
    assert alpha[0] == pytest.approx(1.0)
    assert alpha[-1] < 0.01
    assert np.all(np.diff(alpha) <= 1e-12)
    assert np.all((alpha > 0) & (alpha <= 1))


def test_forward_diffusion_converges_to_the_marginal():
    marginal = np.array([0.1, 0.2, 0.3, 0.4])
    diffusion = DiscreteDiffusion(marginal, n_steps=60)
    rng = np.random.default_rng(0)
    x0 = np.zeros(4000, dtype=int)
    at_zero = diffusion.sample_q(x0, 0, rng)
    assert np.mean(at_zero == 0) > 0.99                 # t=0 leaves data intact
    at_end = diffusion.sample_q(x0, 60, rng)
    observed = np.bincount(at_end, minlength=4) / len(at_end)
    np.testing.assert_allclose(observed, marginal, atol=0.05)


def test_posterior_is_a_probability_distribution():
    marginal = np.array([0.25, 0.25, 0.25, 0.25])
    diffusion = DiscreteDiffusion(marginal, n_steps=50)
    rng = np.random.default_rng(0)
    xt = rng.integers(0, 4, 20)
    probs = rng.dirichlet(np.ones(4), size=20)
    posterior = diffusion.posterior(xt, probs, 25)
    assert posterior.shape == (20, 4)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(posterior >= 0)


def test_posterior_at_t1_returns_the_prediction():
    diffusion = DiscreteDiffusion(np.ones(3) / 3, n_steps=10)
    probs = np.array([[0.7, 0.2, 0.1]])
    np.testing.assert_allclose(diffusion.posterior(np.array([0]), probs, 1), probs)


@pytest.mark.parametrize("smi", ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
                                 "c1ccc2ccccc2c1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"])
def test_graph_encoding_round_trips(smi):
    mol = parse_smiles(smi)
    graph = molecule_to_graph(mol)
    assert graph is not None
    decoded = graph_to_molecule(*graph)
    assert decoded is not None
    assert decoded.formula() == mol.formula()
    assert decoded.n_rings == mol.n_rings


def test_encoding_rejects_out_of_vocabulary_atoms():
    assert molecule_to_graph(parse_smiles("[Se]CC")) is None


def test_decoding_drops_padding_atoms():
    atoms = np.array([0, 1, 1, 0])          # two padding, two carbon
    bonds = np.zeros((4, 4), dtype=int)
    bonds[1, 2] = bonds[2, 1] = 1
    mol = graph_to_molecule(atoms, bonds)
    assert mol is not None and mol.n_atoms == 2 and mol.formula() == "C2H6"


def test_ddpm_marginals_and_sampling_shapes():
    mols = [parse_smiles(s) for s in ("CCO", "CCC", "c1ccccc1", "CC(=O)O")]
    graphs = [molecule_to_graph(m) for m in mols]
    model = MoleculeDDPM(n_steps=5, hidden=16, depth=1, max_atoms=10, seed=0)
    model.fit_marginals(graphs)
    assert model.atom_marginal.shape == (len(ATOM_VOCAB),)
    assert model.bond_marginal.shape == (len(BOND_VOCAB),)
    assert model.atom_marginal.sum() == pytest.approx(1.0)
    atoms, bonds = model.sample(n_atoms=6, seed=0)
    assert atoms.shape == (6,) and bonds.shape == (6, 6)
    np.testing.assert_array_equal(bonds, bonds.T)        # symmetric
    assert np.all(np.diag(bonds) == 0)                   # no self-bonds


def test_ddpm_loss_is_finite_and_differentiable():
    mol = parse_smiles("CCO")
    graphs = [molecule_to_graph(mol)]
    model = MoleculeDDPM(n_steps=5, hidden=16, depth=1, max_atoms=10, seed=0)
    model.fit_marginals(graphs)
    rng = np.random.default_rng(0)
    loss = model.loss(*graphs[0], t=2, rng=rng)
    assert np.isfinite(float(loss.data))
    loss.backward()
    assert any(p.grad is not None for p in model.model.parameters())
