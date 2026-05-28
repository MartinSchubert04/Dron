import { useState } from "react";
import type { TrimValues } from "../hooks/useTrim";

const LIMIT = 20;
const STEP  = 1;

interface AxisTrimProps {
  label:    string;
  hint:     string;
  value:    number;
  onChange: (v: number) => void;
}

function AxisTrim({ label, hint, value, onChange }: AxisTrimProps) {
  const clamp = (v: number) => Math.max(-LIMIT, Math.min(LIMIT, v));

  const barPct = ((value + LIMIT) / (LIMIT * 2)) * 100;
  const barColor =
    value === 0 ? "bg-frame" :
    Math.abs(value) > 10 ? "bg-warn" : "bg-accent";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-white text-[0.65rem] tracking-widest">{label}</span>
          <span className="text-muted text-[0.5rem]">{hint}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => onChange(clamp(value - STEP))}
            className="w-6 h-6 flex items-center justify-center border border-frame rounded
                       text-muted hover:border-accent hover:text-accent text-sm font-mono
                       transition-all active:scale-95"
          >−</button>

          <span className={`w-8 text-center text-[0.75rem] font-mono tabular-nums
            ${value === 0 ? "text-muted" : value > 0 ? "text-accent" : "text-warn"}`}>
            {value > 0 ? `+${value}` : value}
          </span>

          <button
            onClick={() => onChange(clamp(value + STEP))}
            className="w-6 h-6 flex items-center justify-center border border-frame rounded
                       text-muted hover:border-accent hover:text-accent text-sm font-mono
                       transition-all active:scale-95"
          >+</button>

          <button
            onClick={() => onChange(0)}
            disabled={value === 0}
            className="w-6 h-6 flex items-center justify-center border border-frame rounded
                       text-muted hover:border-danger hover:text-danger text-[0.55rem] font-mono
                       transition-all active:scale-95 disabled:opacity-30 disabled:cursor-default"
            title="Reset"
          >✕</button>
        </div>
      </div>

      {/* Mini progress bar — center = 0, left/right = neg/pos */}
      <div className="h-1 bg-frame rounded-full overflow-hidden relative">
        {/* Center marker */}
        <div className="absolute top-0 bottom-0 w-px bg-muted/60" style={{ left: "50%" }} />
        {/* Fill from center */}
        {value !== 0 && (
          <div
            className={`absolute top-0 bottom-0 rounded-full ${barColor} transition-all`}
            style={
              value > 0
                ? { left: "50%", width: `${barPct - 50}%` }
                : { left: `${barPct}%`, width: `${50 - barPct}%` }
            }
          />
        )}
      </div>
    </div>
  );
}

interface Props {
  trim:     TrimValues;
  saving:   boolean;
  saveMsg:  string;
  onChange: (t: TrimValues) => void;
  onSave:   (t: TrimValues) => void;
}

export function TrimPanel({ trim, saving, saveMsg, onChange, onSave }: Props) {
  const [local, setLocal] = useState<TrimValues>(trim);

  const update = (axis: keyof TrimValues, value: number) => {
    const next = { ...local, [axis]: value };
    setLocal(next);
    onChange(next);
  };

  const allZero = Object.values(local).every(v => v === 0);

  return (
    <div className="flex flex-col gap-3 border border-frame rounded-md p-4"
         style={{ background: 'rgba(10,10,18,0.4)' }}>

      <div className="text-accent text-[0.6rem] tracking-[3px] border-b border-frame pb-2">
        TRIM — CORRECCIÓN DE DRIFT
      </div>

      <p className="text-muted text-[0.52rem] -mt-1">
        Ajustá el offset del centro de cada eje para compensar deriva. Rango ±{LIMIT}.
        Guardá para que persista al reiniciar.
      </p>

      <AxisTrim label="ROLL"     hint="Deriva lateral (izq/der)"     value={local.roll}     onChange={v => update("roll",     v)} />
      <AxisTrim label="PITCH"    hint="Deriva adelante/atrás"         value={local.pitch}    onChange={v => update("pitch",    v)} />
      <AxisTrim label="YAW"      hint="Rotación espontánea"           value={local.yaw}      onChange={v => update("yaw",      v)} />
      <AxisTrim label="THROTTLE" hint="Subida/bajada en hover"        value={local.throttle} onChange={v => update("throttle", v)} />

      <div className="flex items-center justify-between pt-2 border-t border-frame">
        <button
          onClick={() => {
            const zero: TrimValues = { roll: 0, pitch: 0, yaw: 0, throttle: 0 };
            setLocal(zero);
            onChange(zero);
          }}
          disabled={allZero}
          className="px-3 py-1.5 text-[0.6rem] border border-warn text-warn rounded
                     hover:bg-warn/10 disabled:opacity-30 disabled:cursor-not-allowed tracking-widest"
        >
          RESET TODO
        </button>

        <span className={`text-[0.55rem] tracking-widest flex-1 text-center mx-2
          ${saveMsg.startsWith("Error") ? "text-danger" : "text-success"}`}>
          {saveMsg}
        </span>

        <button
          onClick={() => onSave(local)}
          disabled={saving}
          className="px-4 py-1.5 text-[0.62rem] border border-success text-success rounded
                     hover:bg-success/10 disabled:opacity-50 tracking-widest font-mono
                     transition-all"
        >
          {saving ? "GUARDANDO…" : "GUARDAR"}
        </button>
      </div>
    </div>
  );
}
