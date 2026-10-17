/**
 * Types for the NEXUS molecular analysis service.
 *
 * The service is the Python package in this repository, started with
 * `python -m nexus.server`. It speaks plain JSON over HTTP and holds no state
 * between requests.
 */

export interface NexusDescriptors {
    mw: number | null;
    heavy_atoms: number | null;
    logp: number | null;
    tpsa: number | null;
    hbd: number | null;
    hba: number | null;
    rotatable_bonds: number | null;
    rings: number | null;
    aromatic_rings: number | null;
    fraction_csp3: number | null;
    drug_likeness: number | null;
    synthetic_accessibility: number | null;
    [key: string]: number | null;
}

export interface NexusMolecule {
    smiles: string;
    formula: string;
    n_atoms: number;
    descriptors: NexusDescriptors;
    lipinski_violations: number;
    veber_pass: boolean;
    structural_alerts: number;
    scaffold: string;
    valence_ok: boolean;
    functional_groups?: Record<string, number>;
}

export interface NexusDescriptorSummary {
    mean: number;
    std: number;
    min: number;
    max: number;
}

export interface NexusAnalyzeResponse {
    n_analyzed: number;
    n_failed: number;
    failed_indices: number[];
    molecules: NexusMolecule[];
    summary: Record<string, NexusDescriptorSummary>;
    elapsed_seconds: number;
}

export interface NexusScaffoldGroup {
    scaffold: string;
    count: number;
    members: string[];
}

export interface NexusScaffoldResponse {
    n_molecules: number;
    n_scaffolds: number;
    n_failed: number;
    groups: NexusScaffoldGroup[];
    elapsed_seconds: number;
}

export interface NexusSimilarityHit {
    smiles: string;
    tanimoto: number;
    index: number;
}

export interface NexusSimilarityResponse {
    query: string;
    n_library: number;
    n_failed: number;
    hits: NexusSimilarityHit[];
    elapsed_seconds: number;
}

export interface NexusPrediction {
    smiles: string;
    prediction: number;
    std: number;
    /** Regression only: conformal interval endpoints. */
    interval_low?: number;
    interval_high?: number;
    /** Classification only: the conformal prediction set. */
    prediction_set?: number[];
    ambiguous?: boolean;
}

export interface NexusPredictResponse {
    task: 'regression' | 'classification';
    alpha: number;
    nominal_coverage: number;
    calibration_coverage: number;
    /** Which conformal score produced the intervals; null for classification. */
    interval_mode: 'constant-width' | 'adaptive' | null;
    /** Mean interval width, or null when the task has no intervals. */
    mean_interval_width: number | null;
    n_labelled: number;
    n_calibration: number;
    n_predicted: number;
    predictions: NexusPrediction[];
    elapsed_seconds: number;
}

export interface NexusBatchItem {
    smiles: string;
    index: number;
    score: number;
}

export interface NexusScreenResponse {
    strategy: string;
    batch_size: number;
    n_measured: number;
    n_pool: number;
    rationale: string;
    n_failed: number;
    batch: NexusBatchItem[];
    elapsed_seconds: number;
}

export interface NexusTopologyMolecule {
    smiles: string;
    n_rings: number;
    ollivier_mean: number;
    ollivier_min: number;
    curvature_features: number[];
    conformer_energy?: number;
    h0_count?: number;
    h1_count?: number;
    h1_total?: number;
    h1_max?: number;
    [key: string]: number | string | number[] | undefined;
}

export interface NexusTopologyResponse {
    n_analyzed: number;
    n_failed: number;
    molecules: NexusTopologyMolecule[];
    elapsed_seconds: number;
}

export interface NexusHealthResponse {
    status: string;
    version: string;
    capabilities: string[];
    max_molecules: number;
}

/** Acquisition strategies the screening endpoint accepts. */
export const NEXUS_STRATEGIES = ['ei', 'ucb', 'greedy', 'uncertainty', 'thompson'] as const;
export type NexusStrategy = (typeof NEXUS_STRATEGIES)[number];

/** Descriptors offered as map colouring options, with display labels. */
export const NEXUS_COLOUR_PROPERTIES: { key: string; label: string }[] = [
    { key: 'drug_likeness', label: 'Drug-likeness' },
    { key: 'mw', label: 'Molecular weight' },
    { key: 'logp', label: 'logP' },
    { key: 'tpsa', label: 'TPSA' },
    { key: 'fraction_csp3', label: 'Fraction sp3' },
    { key: 'synthetic_accessibility', label: 'Synthetic accessibility' },
    { key: 'rings', label: 'Ring count' },
    { key: 'rotatable_bonds', label: 'Rotatable bonds' },
];
