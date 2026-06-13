import { useEffect, useRef } from 'react';
import { init3D, updateAngles, updateMotors, resize3D, dispose3D } from '../drone3d';
import type { TelemetryData } from '../hooks/useTelemetry';
import { useSettings } from '../context/SettingsContext';
import type { AxisInvert } from '../context/SettingsContext';

interface Props {
  telemetry: TelemetryData;
}

export function DroneView3D({ telemetry }: Props) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inited       = useRef(false);

  const { roll, pitch, imu_ok, esp32_ok, gps_ok } = telemetry;
  const { axisInvert, updateSettings } = useSettings();

  useEffect(() => {
    if (!canvasRef.current) return;
    init3D(canvasRef.current);
    inited.current = true;

    const obs = new ResizeObserver(() => {
      if (canvasRef.current) resize3D(canvasRef.current);
    });
    if (containerRef.current) obs.observe(containerRef.current);

    return () => {
      obs.disconnect();
      dispose3D();
      inited.current = false;
    };
  }, []);

  useEffect(() => {
    if (inited.current) updateAngles(
      roll,
      pitch,
      axisInvert.roll     ? -1 : 1,
      axisInvert.pitch    ? -1 : 1,
    );
  }, [roll, pitch, axisInvert.roll, axisInvert.pitch]);

  useEffect(() => {
    if (inited.current) updateMotors([60, 60, 60, 60]);
  }, []);

  const toggleAxis = (key: keyof AxisInvert) =>
    updateSettings({ axisInvert: { ...axisInvert, [key]: !axisInvert[key] } });

  const imuColor = imu_ok ? 'text-success' : 'text-danger';

  return (
    <div ref={containerRef}
         className="flex flex-1 flex-col relative border border-frame rounded-md overflow-hidden min-h-0"
         style={{ background: '#0a0a12' }}>

      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

      {/* HUD — ángulos + estado IMU */}
      <div className="absolute top-3 left-3 flex flex-col gap-1.5 pointer-events-none select-none">
        <HudRow label="ROLL"  value={`${roll.toFixed(1)}°`} />
        <HudRow label="PITCH" value={`${pitch.toFixed(1)}°`} />
        <HudRow label="IMU"   value={imu_ok ? 'OK' : 'FALLO'} valueClass={imuColor} />
        <HudRow label="GPS"   value={gps_ok ? 'OK' : 'FALLO'} valueClass={gps_ok ? 'text-success' : 'text-danger'} />
        <HudRow label="ESP32" value={esp32_ok ? 'Online' : 'Offline'}
                valueClass={esp32_ok ? 'text-success' : 'text-danger'} />
      </div>

      {/* Botones inversión de ejes */}
      <div className="absolute bottom-3 left-3 flex gap-1.5 select-none">
        {(['throttle', 'yaw', 'pitch', 'roll'] as (keyof AxisInvert)[]).map(key => {
          const active = axisInvert[key];
          return (
            <button
              key={key}
              onClick={() => toggleAxis(key)}
              className={`px-2 py-1 rounded text-[0.5rem] font-mono tracking-widest border transition-all
                ${active
                  ? 'border-accent bg-accent/20 text-accent'
                  : 'border-frame/60 bg-black/30 text-muted hover:border-white/30 hover:text-white'}`}
            >
              {key === 'throttle' ? 'THR' : key.toUpperCase()}
              {active && <span className="ml-1 opacity-70">↕</span>}
            </button>
          );
        })}
      </div>

      {/* Leyenda orientación */}
      <div className="absolute bottom-3 right-3 flex items-center gap-2 pointer-events-none select-none
                      text-muted text-[0.48rem] tracking-widest">
        <span className="inline-block w-2 h-2 rounded-sm bg-danger flex-shrink-0" />
        FRENTE
        <span className="mx-1 text-frame">|</span>
        <span className="inline-block w-2 h-2 rounded-sm bg-accent flex-shrink-0" />
        ATRÁS
      </div>

      {/* Overlay cuando no hay conexión */}
      {!esp32_ok && (
        <div className="absolute inset-0 flex items-center justify-center
                        bg-black/60 backdrop-blur-sm pointer-events-none">
          <span className="text-muted text-xs tracking-widest">
            ESP32 sin conexión — configurá la URL en Config
          </span>
        </div>
      )}
    </div>
  );
}

function HudRow({
  label, value, valueClass = 'text-accent',
}: {
  label: string; value: string; valueClass?: string;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-muted text-[0.52rem] tracking-widest w-9 flex-shrink-0">{label}</span>
      <span className={`text-[0.68rem] font-mono ${valueClass}`}>{value}</span>
    </div>
  );
}
