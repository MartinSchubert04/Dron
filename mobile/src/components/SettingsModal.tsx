import { useState } from 'react';

interface Props {
  host:    string;
  onSave:  (host: string) => void;
  onClose: () => void;
}

export function SettingsModal({ host, onSave, onClose }: Props) {
  const [val, setVal] = useState(host);

  const save = () => { if (val.trim()) onSave(val.trim()); onClose(); };

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-surface border border-frame/80 rounded-2xl p-6 w-80 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-white font-bold text-sm tracking-widest uppercase mb-5">Settings</h2>

        <label className="block text-muted text-xs uppercase tracking-wider mb-1.5">
          Backend host
        </label>
        <input
          type="text"
          value={val}
          onChange={e => setVal(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') onClose(); }}
          placeholder="192.168.4.1:8000"
          autoFocus
          className="w-full bg-black/40 text-white border border-frame/80 rounded-lg px-3 py-2.5
                     text-sm font-mono focus:outline-none focus:border-accent transition-colors"
        />
        <p className="text-muted text-[0.6rem] mt-1.5 leading-relaxed">
          IP:port del servidor FastAPI. Se guarda entre sesiones.
        </p>

        <div className="flex justify-end gap-2 mt-5">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-muted hover:text-white transition-colors"
          >
            Cancelar
          </button>
          <button
            onClick={save}
            className="px-5 py-2 text-sm bg-accent text-black font-bold rounded-lg
                       active:scale-95 transition-all tracking-wide"
          >
            Guardar
          </button>
        </div>
      </div>
    </div>
  );
}
