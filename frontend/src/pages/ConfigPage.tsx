import { useState } from 'react';
import { useSettings } from '../context/SettingsContext';
import { TrimPanel } from '../components/TrimPanel';
import { useTrim } from '../hooks/useTrim';

interface Props {
  esp32Url:    string;
  setEsp32Url: (url: string) => void;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 border border-frame rounded-md p-4"
         style={{ background: 'rgba(10,10,18,0.4)' }}>
      <div className="text-accent text-[0.6rem] tracking-[3px] border-b border-frame pb-2">
        {title}
      </div>
      {children}
    </div>
  );
}

function Field({
  label, value, onChange, placeholder, hint, onSave,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  hint?: string;
  onSave?: () => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-white text-[0.65rem] tracking-widest">{label}</label>
      <div className="flex gap-2">
        <input
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && onSave?.()}
          placeholder={placeholder}
          spellCheck={false}
          className="flex-1 px-3 py-1.5 bg-transparent border border-frame rounded
                     text-white text-[0.7rem] font-mono outline-none
                     focus:border-accent transition-colors"
        />
        {onSave && (
          <button
            onClick={onSave}
            className="px-4 py-1.5 text-[0.65rem] border border-accent text-accent rounded
                       hover:bg-accent/10 tracking-widest font-mono transition-all"
          >
            GUARDAR
          </button>
        )}
      </div>
      {hint && <span className="text-muted text-[0.52rem]">{hint}</span>}
    </div>
  );
}

export function ConfigPage({ esp32Url, setEsp32Url }: Props) {
  const { settings, updateSettings } = useSettings();
  const [localHost, setLocalHost]   = useState(settings.backendHost);
  const [localEsp32, setLocalEsp32] = useState(esp32Url);
  const { trim, setTrimState, save: saveTrim, saving, saveMsg } = useTrim();

  const saveHost = () => {
    updateSettings({ backendHost: localHost.trim() });
  };

  const saveEsp32 = () => {
    setEsp32Url(localEsp32.trim());
  };

  return (
    <div className="flex flex-col gap-4 p-4 max-w-xl mx-auto w-full">

      <span className="text-accent text-[0.6rem] tracking-[4px] font-bold">
        CONFIGURACIÓN
      </span>

      <Section title="BACKEND (TURBODRONE API)">
        <Field
          label="Host"
          value={localHost}
          onChange={setLocalHost}
          placeholder="localhost:8000"
          hint="Dirección y puerto del servidor FastAPI. Ej: 192.168.1.10:8000"
          onSave={saveHost}
        />
      </Section>

      <Section title="ESP32-C3 (TELEMETRÍA IMU)">
        <Field
          label="URL completa del endpoint de telemetría"
          value={localEsp32}
          onChange={setLocalEsp32}
          placeholder="http://192.168.1.xx/telemetry"
          hint="El backend hace fetch a esta URL (JSON con roll, pitch, imu_ok). Dejá vacío para deshabilitar."
          onSave={saveEsp32}
        />

        <div className="border border-frame rounded p-3 bg-surface/40">
          <p className="text-muted text-[0.52rem] leading-relaxed">
            <span className="text-accent">Formato esperado del ESP32:</span><br />
            <span className="font-mono text-white">
              {'{ "roll": 3.4, "pitch": -1.2, "imu_ok": true }'}
            </span>
            <br /><br />
            La API hace polling a 10 Hz. Si el ESP32 tarda más de 800 ms en responder,
            la lectura se marca como desconectada.
          </p>
        </div>
      </Section>

      <TrimPanel
        trim={trim}
        saving={saving}
        saveMsg={saveMsg}
        onChange={setTrimState}
        onSave={saveTrim}
      />

      <Section title="ARQUITECTURA">
        <div className="text-muted text-[0.55rem] leading-relaxed space-y-1">
          <div className="flex gap-2 items-start">
            <span className="text-accent flex-shrink-0">→</span>
            <span>La laptop conecta al WiFi del drone (red del drone)</span>
          </div>
          <div className="flex gap-2 items-start">
            <span className="text-accent flex-shrink-0">→</span>
            <span>El backend (FastAPI) también puede alcanzar la red de casa por el adaptador WiFi secundario</span>
          </div>
          <div className="flex gap-2 items-start">
            <span className="text-accent flex-shrink-0">→</span>
            <span>El ESP32-C3 está en la red de casa y expone telemetría IMU vía HTTP</span>
          </div>
          <div className="flex gap-2 items-start">
            <span className="text-accent flex-shrink-0">→</span>
            <span>El backend actúa de proxy: fetchea el ESP32 y expone <span className="font-mono text-white">GET /telemetry</span></span>
          </div>
        </div>
      </Section>

    </div>
  );
}
