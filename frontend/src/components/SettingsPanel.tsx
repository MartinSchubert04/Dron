import { useState } from 'react';
import { useSettings } from '../context/SettingsContext';

interface Props {
  onClose: () => void;
}

export function SettingsPanel({ onClose }: Props) {
  const { settings, updateSettings } = useSettings();
  const [host, setHost] = useState(settings.backendHost);

  const save = () => {
    const trimmed = host.trim();
    if (trimmed) updateSettings({ backendHost: trimmed });
    onClose();
  };

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-white/10 rounded-xl p-6 w-80 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-white font-bold text-base mb-5 tracking-widest uppercase">Settings</h2>

        <label className="block text-gray-400 text-xs uppercase tracking-wider mb-1.5">
          Backend host
        </label>
        <input
          type="text"
          value={host}
          onChange={e => setHost(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') onClose(); }}
          placeholder="localhost:8000"
          autoFocus
          className="w-full bg-gray-800 text-white border border-white/10 rounded px-3 py-2 text-sm font-mono
                     focus:outline-none focus:border-blue-500 transition-colors"
        />
        <p className="text-gray-500 text-xs mt-1.5">
          Applies to API calls, WebSocket, and the video feed. Changes take effect immediately.
        </p>

        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-sm text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={save}
            className="px-4 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors font-semibold"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
