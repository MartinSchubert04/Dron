import React from "react";
import type { ControlMode } from "../hooks/useControls";

interface Props {
  mode: ControlMode;
  setMode: (m: ControlMode) => void;
  gamepadConnected: boolean;
}

export const ControlSchemeToggle: React.FC<Props> = ({ mode, setMode, gamepadConnected }) => {
  const toKeyboard = () => setMode("inc");

  const toGamepad = () => {
    if (gamepadConnected) setMode("abs");
  };

  const toTrackPoint = () => {
    document.body.requestPointerLock();
    setMode("mouse");
  };

  const toTouch = () => setMode("touch");

  const btn = (active: boolean, color: string) =>
    `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
      active ? color : "bg-gray-600 hover:bg-gray-500"
    }`;

  return (
    <div className="absolute bottom-4 left-4 z-30 bg-surface/80 backdrop-blur-md
                    border border-frame/80 rounded-lg shadow-xl p-4">
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={toKeyboard}  className={btn(mode === "inc",   "bg-sky-600")}>
            Keyboard
          </button>

          <button
            onClick={toGamepad}
            disabled={!gamepadConnected}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              mode === "abs"
                ? "bg-green-600"
                : gamepadConnected
                  ? "bg-gray-600 hover:bg-gray-500"
                  : "bg-gray-700 cursor-not-allowed opacity-60"
            }`}
          >
            Gamepad
          </button>

          <button onClick={toTrackPoint} className={btn(mode === "mouse", "bg-red-600")}>
            TrackPoint
          </button>

          <button onClick={toTouch} className={btn(mode === "touch", "bg-accent text-black")}>
            Touch
          </button>
        </div>

        <div className="text-xs text-muted text-center">
          {mode === "inc"    && <><span className="text-white font-semibold">Keyboard</span> — WASD / Arrows</>}
          {mode === "abs"    && <><span className="text-white font-semibold">Gamepad</span></>}
          {mode === "mouse"  && <><span className="text-white font-semibold">TrackPoint</span>&nbsp;<span className="text-gray-400">(Esc to release)</span></>}
          {mode === "touch"  && <><span className="text-white font-semibold">Touch</span> — on-screen joysticks</>}
        </div>
      </div>
    </div>
  );
};
