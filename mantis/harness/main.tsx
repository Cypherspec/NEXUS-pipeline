import React from 'react';
import { createRoot } from 'react-dom/client';

import NexusPanel from '@/app/components/panels/NexusPanel';

import { MapPreview } from './stubs/mantisStores';

/** Mantis-like shell: a dark workspace with the panel docked beside the map. */
const App: React.FC = () => (
    <div className="flex h-screen w-screen flex-col" style={{ background: '#050609' }}>
        <div className="flex items-center gap-2 px-3 py-1.5"
            style={{ borderBottom: '1px solid rgba(255,255,255,0.07)', background: '#07090d' }}>
            <span className="text-[11px] font-medium" style={{ color: '#93a4ba' }}>Mantis</span>
            <span className="text-[11px]" style={{ color: '#3c4a5c' }}>/</span>
            <span className="text-[11px]" style={{ color: '#5d6e84' }}>ESOL solubility set</span>
            <span className="ml-auto text-[10px]" style={{ color: '#3c4a5c' }}>
                panel harness
            </span>
        </div>
        <div className="flex min-h-0 flex-1 gap-2 p-2">
            <div className="flex w-[400px] shrink-0 flex-col overflow-hidden rounded-xl"
                style={{ border: '1px solid rgba(255,255,255,0.08)' }}>
                <NexusPanel />
            </div>
            <div className="flex-1 overflow-hidden rounded-xl"
                style={{ border: '1px solid rgba(255,255,255,0.08)' }}>
                <MapPreview />
            </div>
        </div>
    </div>
);

createRoot(document.getElementById('root')!).render(<App />);
