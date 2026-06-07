import AxisIndicator from "./AxisIndicator";
import type { Axes, CameraTiltDirection, CommandCapabilities, ConnState, ControlMode, SpeedTier } from "../hooks/useControls";
// ConnState kept in import for the prop interface
import CommandButtons from "./CommandButtons";
import SpeedControl from "./SpeedControl";
import CameraTiltControl from "./CameraTiltControl";

interface ControlsOverlayProps {
  axes: Axes;
  mode: ControlMode;
  connState: ConnState;
  droneType: string;
  commandCapabilities: CommandCapabilities;
  speedTier: SpeedTier;
  cameraTiltDirection: CameraTiltDirection;
  onTakeoff: () => void;
  onLand: () => void;
  onEstop: () => void;
  onCalibrate: () => void;
  onSpeedChange: (tier: SpeedTier) => void;
  onCameraTiltChange: (direction: CameraTiltDirection) => void;
}


export default function ControlsOverlay({
  axes,
  mode,
  connState,
  droneType,
  commandCapabilities,
  speedTier,
  cameraTiltDirection,
  onTakeoff,
  onLand,
  onEstop,
  onCalibrate,
  onSpeedChange,
  onCameraTiltChange,
}: ControlsOverlayProps) {
  const left      = { x: axes.roll,  y: axes.pitch };
  const right     = { x: axes.yaw,   y: axes.throttle };
  const droneLabel = droneType === "unknown" ? "Unknown" : droneType.replace(/_/g, " ").toUpperCase();

  return (
    <div className="absolute inset-0 pointer-events-none z-10">

      {/* ── Controls area ──────────────────────────────────────────────────── */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2
                      flex flex-col items-center gap-5 z-20 pointer-events-auto">

        <CommandButtons
          droneType={droneLabel}
          capabilities={commandCapabilities}
          onTakeoff={onTakeoff}
          onLand={onLand}
          onEstop={onEstop}
          onCalibrate={onCalibrate}
        />

        {commandCapabilities.speed_control && (
          <SpeedControl
            enabled={commandCapabilities.speed_control}
            value={speedTier}
            onChange={onSpeedChange}
          />
        )}

        {commandCapabilities.camera_tilt && (
          <CameraTiltControl
            activeDirection={cameraTiltDirection}
            onTiltChange={onCameraTiltChange}
          />
        )}

        {/* Sticks — hidden in touch mode (nipple joysticks replace them visually) */}
        {mode !== "touch" && (
          <div className="flex gap-10">
            <AxisIndicator {...left}  label="PITCH / ROLL" />
            <AxisIndicator {...right} label="YAW / THROTTLE" />
          </div>
        )}

        {/* Keyboard hint */}
        {mode === "inc" && (
          <p className="text-[0.48rem] text-muted text-center tracking-wide select-none">
            W/S&nbsp;=&nbsp;Pitch&nbsp;·&nbsp;A/D&nbsp;=&nbsp;Roll&nbsp;·&nbsp;
            ↑↓&nbsp;=&nbsp;Throttle&nbsp;·&nbsp;←→&nbsp;=&nbsp;Yaw&nbsp;·&nbsp;
            PgUp/PgDn&nbsp;=&nbsp;Tilt
          </p>
        )}
      </div>
    </div>
  );
}
