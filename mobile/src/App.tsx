import { useState, useCallback } from "react";
import { useSettings } from "./hooks/useSettings";
import { useDrone } from "./hooks/useDrone";
import type { SpeedTier } from "./hooks/useDrone";
import { useVoiceCommands } from "./hooks/useVoiceCommands";
import { VideoFeed } from "./components/VideoFeed";
import { NippleJoysticks } from "./components/NippleJoysticks";
import { ActionPanel } from "./components/ActionPanel";
import { SettingsModal } from "./components/SettingsModal";

/* ── Icons ─────────────────────────────────────────────────────────────── */
function MenuIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      className="w-5 h-5"
    >
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}
function CloseIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      className="w-5 h-5"
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}
function MicIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-4 h-4"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="17" x2="12" y2="21" />
      <line x1="8" y1="21" x2="16" y2="21" />
    </svg>
  );
}
function GearIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      className="w-5 h-5"
    >
      <circle cx="12" cy="12" r="3" />
      <path
        d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06
               a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09
               A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83
               l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09
               A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83
               l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09
               a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83
               l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09
               a1.65 1.65 0 0 0-1.51 1z"
      />
    </svg>
  );
}

/* ── Connection dot ─────────────────────────────────────────────────────── */
const DOT_CLASS: Record<string, string> = {
  connected: "bg-success shadow-[0_0_5px_#66bb6a]",
  connecting: "bg-warn animate-pulse",
  disconnected: "bg-danger",
};

/* ── App ────────────────────────────────────────────────────────────────── */
export default function App() {
  const { settings, update, apiBase, wsBase } = useSettings();
  const drone = useDrone(apiBase, wsBase);

  const [panelOpen, setPanelOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const togglePanel = () => setPanelOpen((o) => !o);

  const handleTiltStart = useCallback(
    (dir: -1 | 1) => drone.setCameraTilt(dir),
    [drone],
  );
  const handleTiltEnd = useCallback(() => drone.setCameraTilt(0), [drone]);

  const {
    listening,
    supported: voiceSupported,
    toggle: toggleVoice,
    lastCommand,
  } = useVoiceCommands({
    onTakeoff: drone.takeoff,
    onLand: drone.land,
    onSpeed: drone.setSpeed,
  });

  // Close the action panel when tapping outside it (on the video/backdrop)
  const closePanel = () => {
    if (panelOpen) setPanelOpen(false);
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-black select-none">
      {/* ── Video background ─────────────────────────────────────────── */}
      <VideoFeed src={`${apiBase}/mjpeg`} />

      {/* ── Backdrop tap-to-close when panel open ────────────────────── */}
      {panelOpen && (
        <div className="absolute inset-0 z-10" onClick={closePanel} />
      )}

      {/* ── Top bar ──────────────────────────────────────────────────── */}
      <div
        className="absolute top-0 inset-x-0 h-14 z-30 flex items-center justify-between
                      px-3 bg-black/60 backdrop-blur-sm border-b border-white/10"
        style={{ paddingTop: "env(safe-area-inset-top, 0px)" }}
      >
        {/* Left: menu toggle */}
        <button
          onPointerDown={togglePanel}
          className={`w-10 h-10 flex items-center justify-center rounded-lg transition-colors
            ${
              panelOpen ?
                "bg-accent/20 text-accent border border-accent/40"
              : "text-white/70 hover:text-white hover:bg-white/10"
            }`}
        >
          {panelOpen ?
            <CloseIcon />
          : <MenuIcon />}
        </button>

        {/* Center: title + connection */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${DOT_CLASS[drone.connState]}`}
          />
          <span className="text-white font-bold text-sm tracking-[0.2em]">
            TURBODRONE
          </span>
          <span className="text-muted text-[0.55rem] tracking-wide">
            {drone.droneType !== "unknown" ?
              drone.droneType.replace(/_/g, " ").toUpperCase()
            : ""}
          </span>
        </div>

        {/* Right: voice + settings */}
        <div className="flex items-center gap-1">
          {voiceSupported && (
            <button
              onPointerDown={toggleVoice}
              className={`flex items-center gap-1 px-2 py-1.5 rounded-full border
                          text-[0.55rem] font-mono tracking-widest uppercase transition-all
                          ${
                            listening ?
                              "bg-danger/20 border-danger text-danger animate-pulse"
                            : "border-white/20 text-white/50"
                          }`}
            >
              <MicIcon />
              {listening ? "on" : ""}
            </button>
          )}
          <button
            onPointerDown={() => setSettingsOpen(true)}
            className="w-10 h-10 flex items-center justify-center text-white/50
                       hover:text-white rounded-lg hover:bg-white/10 transition-colors"
          >
            <GearIcon />
          </button>
        </div>
      </div>

      {/* ── Voice command toast ──────────────────────────────────────── */}
      {lastCommand && (
        <div
          className="fixed top-16 right-4 z-40 px-3 py-1.5 rounded-lg pointer-events-none
                        text-[0.65rem] font-mono tracking-wider select-none
                        bg-black/80 border border-accent/50 text-accent"
        >
          ✓ {lastCommand}
        </div>
      )}

      {/* ── Action panel (slide-down) ────────────────────────────────── */}
      <ActionPanel
        open={panelOpen}
        capabilities={drone.capabilities}
        speedTier={drone.speedTier}
        onTakeoff={() => {
          drone.takeoff();
          setPanelOpen(false);
        }}
        onLand={() => {
          drone.land();
          setPanelOpen(false);
        }}
        onEstop={() => {
          drone.estop();
          setPanelOpen(false);
        }}
        onSpeed={(t: SpeedTier) => drone.setSpeed(t)}
        onTiltStart={handleTiltStart}
        onTiltEnd={handleTiltEnd}
      />

      {/* ── Joysticks ────────────────────────────────────────────────── */}
      <NippleJoysticks onLeft={drone.setLeft} onRight={drone.setRight} />

      {/* ── Settings modal ───────────────────────────────────────────── */}
      {settingsOpen && (
        <SettingsModal
          host={settings.backendHost}
          onSave={(host) => update({ backendHost: host })}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}
