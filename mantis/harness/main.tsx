import React from 'react';
import { createRoot } from 'react-dom/client';

import NexusPanel from '@/app/components/panels/NexusPanel';

import { MapPreview } from './stubs/mantisStores';

const App: React.FC = () => (
    <div className="flex h-screen w-screen gap-3 bg-[#f9f9f7] p-3">
        <div className="flex w-[380px] shrink-0 flex-col overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
            <NexusPanel />
        </div>
        <div className="flex-1 overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
            <MapPreview />
        </div>
    </div>
);

createRoot(document.getElementById('root')!).render(<App />);
