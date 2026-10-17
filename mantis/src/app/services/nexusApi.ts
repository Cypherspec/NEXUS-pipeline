/**
 * Client for the NEXUS molecular analysis service.
 *
 * The base URL comes from NEXT_PUBLIC_NEXUS_URL and falls back to the local
 * default port, so a developer running `python -m nexus.server` needs no
 * further configuration.
 */

import type {
    NexusAnalyzeResponse,
    NexusHealthResponse,
    NexusPredictResponse,
    NexusScaffoldResponse,
    NexusScreenResponse,
    NexusSimilarityResponse,
    NexusStrategy,
    NexusTopologyResponse,
} from '@/types/nexus';

const DEFAULT_URL = 'http://127.0.0.1:8765';
const DEFAULT_TIMEOUT_MS = 120_000;

export function nexusBaseUrl(): string {
    const configured = process.env.NEXT_PUBLIC_NEXUS_URL;
    return (configured && configured.trim()) || DEFAULT_URL;
}

export class NexusServiceError extends Error {
    constructor(message: string, readonly status?: number) {
        super(message);
        this.name = 'NexusServiceError';
    }
}

async function request<T>(path: string, body?: unknown,
                          signal?: AbortSignal): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
    // Abort on either the caller's signal or our own timeout.
    signal?.addEventListener('abort', () => controller.abort(), { once: true });

    let response: Response;
    try {
        response = await fetch(`${nexusBaseUrl()}${path}`, {
            method: body === undefined ? 'GET' : 'POST',
            headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
            body: body === undefined ? undefined : JSON.stringify(body),
            signal: controller.signal,
        });
    } catch (error) {
        if ((error as Error).name === 'AbortError') {
            throw new NexusServiceError(
                signal?.aborted ? 'Request cancelled.'
                    : `NEXUS did not respond within ${DEFAULT_TIMEOUT_MS / 1000}s.`);
        }
        throw new NexusServiceError(
            `Could not reach the NEXUS service at ${nexusBaseUrl()}. `
            + 'Start it with: python -m nexus.server');
    } finally {
        clearTimeout(timeout);
    }

    const text = await response.text();
    let payload: unknown;
    try {
        payload = text ? JSON.parse(text) : {};
    } catch {
        throw new NexusServiceError(
            `NEXUS returned a non-JSON response (HTTP ${response.status}).`,
            response.status);
    }
    if (!response.ok) {
        const message = (payload as { error?: string })?.error
            ?? `NEXUS request failed (HTTP ${response.status}).`;
        throw new NexusServiceError(message, response.status);
    }
    return payload as T;
}

export function fetchNexusHealth(signal?: AbortSignal): Promise<NexusHealthResponse> {
    return request<NexusHealthResponse>('/health', undefined, signal);
}

export function analyzeMolecules(
    smiles: string[], options: { functionalGroups?: boolean } = {},
    signal?: AbortSignal,
): Promise<NexusAnalyzeResponse> {
    return request<NexusAnalyzeResponse>(
        '/analyze', { smiles, functional_groups: options.functionalGroups ?? false }, signal);
}

export function groupByScaffold(smiles: string[], generic = false,
                                signal?: AbortSignal): Promise<NexusScaffoldResponse> {
    return request<NexusScaffoldResponse>('/scaffolds', { smiles, generic }, signal);
}

export function searchSimilar(query: string, library: string[], topK = 50,
                              signal?: AbortSignal): Promise<NexusSimilarityResponse> {
    return request<NexusSimilarityResponse>(
        '/similarity', { query, library, top_k: topK }, signal);
}

export function predictWithIntervals(
    smiles: string[], labels: (number | null)[], alpha = 0.1,
    options: { adaptive?: boolean } = {}, signal?: AbortSignal,
): Promise<NexusPredictResponse> {
    return request<NexusPredictResponse>('/predict', {
        smiles, labels, alpha, adaptive: options.adaptive ?? false,
    }, signal);
}

export function proposeScreeningBatch(
    smiles: string[], measured: Record<string, number>,
    options: { batchSize?: number; strategy?: NexusStrategy; diversity?: number } = {},
    signal?: AbortSignal,
): Promise<NexusScreenResponse> {
    return request<NexusScreenResponse>('/screen', {
        smiles,
        measured,
        batch_size: options.batchSize ?? 32,
        strategy: options.strategy ?? 'ei',
        diversity: options.diversity ?? 0,
    }, signal);
}

export function computeTopology(smiles: string[], conformer = true,
                                signal?: AbortSignal): Promise<NexusTopologyResponse> {
    return request<NexusTopologyResponse>('/topology', { smiles, conformer }, signal);
}
