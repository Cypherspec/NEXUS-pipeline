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
    bg: '#0f1014',                                  // --workspace-canvas
    surface: '#191a1f',                             // --workspace-panel
    surfaceInset: '#24262c',                        // --workspace-panel-raised
    line: '#2b2d35',                                // --workspace-border
    lineStrong: '#434652',                          // --workspace-border-strong
    text: '#eeeef2',                                // --workspace-foreground
    text2: '#a2a5b0',                               // --workspace-muted
    text3: '#777b87',                               // between muted and border-strong
    grid: 'rgba(255, 255, 255, 0.06)',
    // Chrome and data marks are not the same job. Mantis's workspace tones
    // are accents for chrome, and as a categorical set on #191a1f they fail
    // the lightness band and sit in the colour-vision warn band (green/amber
    // dE 6.7 protan). Data marks therefore use a set validated against this
    // surface -- worst all-pairs CVD dE 9.4, normal-vision 20.9 -- while the
    // accent, gradient and status colours below stay Mantis's own.
    series: ['#3987e5', '#d95926', '#199e70'],
    // Single hue, dim to bright on a dark surface, anchored on the azure the
    // workspace accents with.
    ramp: ['#17243a', '#1d3a66', '#245593', '#006aff', '#5b9dff', '#8dbaff', '#cbdfff'],
    accent: '#5b9dff',                              // --workspace-accent
    good: '#3ed68c',                                // --workspace-tone-green
    warn: '#ffb84d',                                // --workspace-tone-amber
    crit: '#ff7aa8',                                // --workspace-tone-rose
    gradient: 'linear-gradient(104deg, #8dbaff 0%, #5b9dff 46%, #a990ff 100%)',
    tooltipBg: 'rgba(25, 26, 31, 0.97)',
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
