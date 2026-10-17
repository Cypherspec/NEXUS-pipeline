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
    bg: '#07090d',
    surface: '#0a0d12',
    surfaceInset: '#10151d',
    line: 'rgba(255, 255, 255, 0.07)',
    lineStrong: 'rgba(255, 255, 255, 0.16)',
    text: '#e9eff8',
    text2: '#93a4ba',
    text3: '#5d6e84',
    grid: 'rgba(255, 255, 255, 0.055)',
    series: ['#3987e5', '#d95926', '#199e70'],
    // Monotone increasing lightness, one hue: on a near-black surface the low
    // end must be dim and the high end bright, not the other way round.
    ramp: ['#0f2d52', '#164a85', '#1d68b8', '#3987e5', '#67a9f0', '#9bc9f8', '#d4e8ff'],
    accent: '#4f9dff',
    good: '#199e70',
    warn: '#c98500',
    crit: '#e66767',
    gradient: 'linear-gradient(104deg, #a5dbff 0%, #4f9dff 46%, #7b7cff 100%)',
    tooltipBg: 'rgba(12, 17, 24, 0.96)',
};

const LIGHT: NexusTheme = {
    bg: '#ffffff',
    surface: '#ffffff',
    surfaceInset: '#f5f7fa',
    line: 'rgba(15, 23, 42, 0.1)',
    lineStrong: 'rgba(15, 23, 42, 0.22)',
    text: '#0b1220',
    text2: '#51607a',
    text3: '#7c8aa0',
    grid: 'rgba(15, 23, 42, 0.07)',
    series: ['#2a78d6', '#eb6834', '#1baf7a'],
    ramp: ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281'],
    accent: '#2a78d6',
    good: '#1baf7a',
    warn: '#eda100',
    crit: '#e34948',
    gradient: 'linear-gradient(104deg, #1c5cab 0%, #2a78d6 46%, #5250c7 100%)',
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
