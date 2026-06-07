import { DroneView3D } from '../components/DroneView3D';
import type { TelemetryData } from '../hooks/useTelemetry';

interface Props {
  telemetry: TelemetryData;
}

export function ThreeDPage({ telemetry }: Props) {
  return (
    <div className="flex flex-col h-full p-3 gap-3 overflow-hidden">

      {/* Header row */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <span className="text-accent text-[0.6rem] tracking-[4px] font-bold">
          VISTA 3D / TELEMETRÍA IMU
        </span>
        <span className="flex-1 h-px bg-frame" />
        <div className="flex items-center gap-2 text-[0.6rem] font-mono">
          <span className="text-muted">Roll</span>
          <span className={`${Math.abs(telemetry.roll)  > 10 ? 'text-warn' : 'text-white'}`}>
            {telemetry.roll.toFixed(2)}°
          </span>
          <span className="text-muted ml-2">Pitch</span>
          <span className={`${Math.abs(telemetry.pitch) > 10 ? 'text-warn' : 'text-white'}`}>
            {telemetry.pitch.toFixed(2)}°
          </span>
        </div>
      </div>

      {/* 3D canvas — fills remaining space */}
      <DroneView3D telemetry={telemetry} />

      {/* Info footer */}
      <div className="flex-shrink-0 text-muted text-[0.5rem] tracking-wide text-center">
        Modelo refleja roll y pitch del ESP32-C3 en tiempo real · la cámara no sigue la orientación del modelo
      </div>
    </div>
  );
}
