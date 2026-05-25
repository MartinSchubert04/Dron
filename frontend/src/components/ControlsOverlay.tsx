import AxisIndicator from "./AxisIndicator";
import type { Axes, CameraTiltDirection, CommandCapabilities, ConnState, ControlMode, SpeedTier } from "../hooks/useControls";
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
  onSpeedChange: (tier: SpeedTier) => void;
  onCameraTiltChange: (direction: CameraTiltDirection) => void;
}

const CONN_DOT: Record<ConnState, string> = {
  connected:    "bg-success shadow-[0_0_6px_#66bb6a]",
  connecting:   "bg-warn animate-pulse",
  disconnected: "bg-danger",
};

const CONN_LABEL: Record<ConnState, string> = {
  connected:    "Online",
  connecting:   "Connecting…",
  disconnected: "Offline",
};

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
  onSpeedChange,
  onCameraTiltChange,
}: ControlsOverlayProps) {
  const left      = { x: axes.roll,  y: axes.pitch };
  const right     = { x: axes.yaw,   y: axes.throttle };
  const droneLabel = droneType === "unknown" ? "Unknown" : droneType.replace(/_/g, " ").toUpperCase();

  return (
    <div className="absolute inset-0 pointer-events-none z-10">

      {/* ── Title bar ─────────────────────────────────────────────────────── */}
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between
                      px-5 py-4 bg-black/60 backdrop-blur-sm border-b border-white/10">

        {/* Left: connection status */}
        <div className="flex items-center gap-2 min-w-[110px]">
          <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${CONN_DOT[connState]}`} />
          <span className="text-[0.65rem] text-muted tracking-wide">{CONN_LABEL[connState]}</span>
        </div>

        {/* Center: title */}
        <h1
          className="font-heading font-bold text-white drop-shadow-lg select-none"
          style={{ fontSize: "2rem", letterSpacing: "0.1em", lineHeight: 1.2 }}
        >
          TURBODRONE WEB
        </h1>

        {/* Right: spacer to balance the layout */}
        <div className="min-w-[110px]" />
      </div>

      {/* ── Controls area ──────────────────────────────────────────────────── */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2
                      flex flex-col items-center gap-5 z-20 pointer-events-auto">

        <CommandButtons
          droneType={droneLabel}
          capabilities={commandCapabilities}
          onTakeoff={onTakeoff}
          onLand={onLand}
          onEstop={onEstop}
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
