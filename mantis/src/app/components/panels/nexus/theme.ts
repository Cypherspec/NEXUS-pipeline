/**
 * Visual tokens for the NEXUS panel.
 *
 * Both modes are selected rather than flipped: the dark steps were chosen for
 * a near-black surface and validated as a set against it, not derived from the
 * light ones. The three categorical hues pass the all-pairs colour-vision and
 * normal-vision floors on both surfaces (worst CVD dE 9.4, normal-vision 20.9
 * on #0a0d12), which is why the series cap is three -- a fourth family folds
 * into "other" rather than inventing a hue.
 */

import type React from 'react';

export interface NexusTheme {
    /** Panel backdrop. */
    bg: string;
    /** Raised card surface. */
    surface: string;
    /** Surface for nested wells and chart plots. */
    surfaceInset: string;
    /** Hairline borders. */
    line: string;
    /** Border for an emphasised or selected element. */
    lineStrong: string;
    /** Primary, secondary and muted ink. Values and labels wear these, never a series hue. */
    text: string;
    text2: string;
    text3: string;
    /** Recessive gridlines. */
    grid: string;
    /** Categorical hues, assigned in fixed order and never cycled. */
    series: readonly [string, string, string];
    /** Single-hue sequential ramp, dim to bright on dark, light to dark on light. */
    ramp: readonly string[];
    /**
     * Diverging scale for signed quantities, as two hues around a neutral
     * midpoint. Discrete Ricci curvature is the case this exists for: it is
     * signed, and zero is a meaningful value -- a cycle graph has curvature
     * exactly zero -- so a sequential ramp would hide the thing worth seeing.
     */
    diverging: { low: string; mid: string; high: string };
    /** Accent for interactive chrome. */
    accent: string;
    /** Reserved status colours; never reused as a series. */
    good: string;
    warn: string;
    crit: string;
    /** Headline gradient, the one visual flourish. */
    gradient: string;
    /** Tooltip surface. */
    tooltipBg: string;
}

const DARK: NexusTheme = {
    // These are Mantis's own workspace tokens, copied by value rather than
    // referenced. The panel runs in a sandboxed iframe with its own document,
    // so it inherits none of the host's custom properties; copying keeps the
    // panel in step with the chrome around it, and any drift is a one-line
    // change here against src/styles/design-tokens.css.
    bg: '#0d0e10',                                  // --workspace-canvas
    surface: '#131417',                             // --workspace-panel
    surfaceInset: '#0f1012',                        // --workspace-panel-raised
    line: '#222327',                                // --workspace-border
    lineStrong: '#34363c',                          // --workspace-border-strong
    text: '#eeeef2',                                // --workspace-foreground
    text2: '#9a9ca3',                               // --workspace-muted
    text3: '#6b6e76',                               // between muted and border-strong
    grid: 'rgba(255, 255, 255, 0.06)',
    // Chrome and data marks are not the same job. Mantis's workspace tones
    // are accents for chrome, and as a categorical set on #191a1f they fail
    // the lightness band and sit in the colour-vision warn band (green/amber
    // dE 6.7 protan). Data marks therefore use a set validated against this
    // surface -- worst all-pairs CVD dE 9.4, normal-vision 20.9 -- while the
    // accent, gradient and status colours below stay Mantis's own.
    series: ['#bfe3dc', '#d9b88a', '#9db8d9'],
    // Single hue, dim to bright on a dark surface, anchored on the azure the
    // workspace accents with.
    ramp: ['#24302f', '#2f4744', '#3f625d', '#5a837c', '#7fa8a0', '#acccc5', '#e6f2ef'],
    diverging: { low: '#9fd3c9', mid: '#55585f', high: '#e0b07a' },
    accent: '#e8e8ea',                              // --workspace-accent
    good: '#3ed68c',                                // --workspace-tone-green
    warn: '#ffb84d',                                // --workspace-tone-amber
    crit: '#ff7aa8',                                // --workspace-tone-rose
    gradient: 'linear-gradient(90deg, #f2f2f3 0%, #f2f2f3 100%)',
    tooltipBg: 'rgba(19, 20, 23, 0.97)',
};

const LIGHT: NexusTheme = {
    bg: '#eceef3',                                  // --workspace-canvas
    surface: '#ffffff',                             // --workspace-panel
    surfaceInset: '#f3f5f9',                        // --workspace-panel-raised
    line: '#dfe2ea',                                // --workspace-border
    lineStrong: '#c0c5d1',                          // --workspace-border-strong
    text: '#171717',                                // --workspace-foreground
    text2: '#5f636e',                               // --workspace-muted
    text3: '#8b8f9a',
    grid: 'rgba(23, 23, 23, 0.07)',
    series: ['#006aff', '#d17a00', '#0a9a4e'],
    ramp: ['#dce8ff', '#b6d0ff', '#84b2ff', '#3f8bff', '#006aff', '#0054cc', '#003a8c'],
    diverging: { low: '#2f6fdb', mid: '#8b8f9a', high: '#d1407a' },
    accent: '#006aff',                              // --mantis-azure
    good: '#0a9a4e',
    warn: '#d17a00',
    crit: '#e11d6a',
    gradient: 'linear-gradient(104deg, #0054cc 0%, #006aff 46%, #7c4dff 100%)',
    tooltipBg: 'rgba(255, 255, 255, 0.97)',
};

/**
 * Returns the token set for the active mode.
 *
 * @param isDarkMode Whether Mantis is in dark mode.
 * @returns The matching theme.
 */
export function nexusTheme(isDarkMode: boolean): NexusTheme {
    return isDarkMode ? DARK : LIGHT;
}

/**
 * Maps a value onto the sequential ramp.
 *
 * @param value Value to colour.
 * @param min Lower bound of the scale.
 * @param max Upper bound of the scale.
 * @param ramp Ramp to sample.
 * @returns A hex colour.
 */
export function rampColor(value: number, min: number, max: number,
                          ramp: readonly string[]): string {
    if (!Number.isFinite(value) || max <= min) return ramp[0];
    const unit = Math.min(1, Math.max(0, (value - min) / (max - min)));
    return ramp[Math.min(ramp.length - 1, Math.round(unit * (ramp.length - 1)))];
}

/** Style that paints text with the headline gradient. */
export function gradientText(theme: NexusTheme): React.CSSProperties {
    return {
        backgroundImage: theme.gradient,
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
    };
}


/**
 * Maps a signed value onto the diverging scale.
 *
 * The midpoint is fixed at zero rather than at the data's middle, because for
 * a signed quantity the sign is the information: a scale centred on the mean
 * would paint a wholly positive set as if it straddled zero.
 *
 * @param value Signed value to colour.
 * @param magnitude Largest absolute value in the set, used to normalise.
 * @param theme Active theme.
 * @returns A CSS colour.
 */
export function divergingColor(value: number, magnitude: number,
                               theme: NexusTheme): string {
    if (!Number.isFinite(value) || magnitude <= 0) return theme.diverging.mid;
    const unit = Math.max(-1, Math.min(1, value / magnitude));
    const to = unit >= 0 ? theme.diverging.high : theme.diverging.low;
    return mixHex(theme.diverging.mid, to, Math.abs(unit));
}

/**
 * Blends two hex colours in sRGB.
 *
 * @param from Start colour.
 * @param to End colour.
 * @param amount Fraction of `to`, in [0, 1].
 * @returns The blended hex colour.
 */
function mixHex(from: string, to: string, amount: number): string {
    const parse = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
    const a = parse(from);
    const b = parse(to);
    const mixed = a.map((channel, i) => Math.round(channel + (b[i] - channel) * amount));
    return `#${mixed.map(v => v.toString(16).padStart(2, '0')).join('')}`;
}
