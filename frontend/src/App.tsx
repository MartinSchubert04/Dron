import { useState } from 'react';
import { useControls } from './hooks/useControls';
import { useSettings } from './context/SettingsContext';
import { ControlSchemeToggle } from './components/ControlSchemeToggle';
import { VideoFeed } from './components/VideoFeed';
import ControlsOverlay from './components/ControlsOverlay';
import { PluginControls } from './components/PluginControls';
import { DrawingOverlay } from './components/DrawingOverlay';
import { SettingsPanel } from './components/SettingsPanel';
import { NippleJoysticks } from './components/NippleJoysticks';

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
         strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06
               a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09
               A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83
               l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09
               A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83
               l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09
               a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83
               l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09
               a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function App() {
  const { apiBase } = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const {
    axes,
    mode,
    setMode,
    connState,
    gamepadConnected,
    droneType,
    commandCapabilities,
    takeOff,
    land,
    emergencyStop,
    speedTier,
    setSpeedTier,
    cameraTiltDirection,
    setCameraTiltDirection,
    setTouchLeft,
    setTouchRight,
  } = useControls();

  return (
    <div className="relative w-screen h-screen bg-black">
      <VideoFeed src={`${apiBase}/mjpeg`} />
      <DrawingOverlay />

      <ControlsOverlay
        axes={axes}
        mode={mode}
        connState={connState}
        droneType={droneType}
        commandCapabilities={commandCapabilities}
        speedTier={speedTier}
        cameraTiltDirection={cameraTiltDirection}
        onTakeoff={takeOff}
        onLand={land}
        onEstop={emergencyStop}
        onSpeedChange={setSpeedTier}
        onCameraTiltChange={setCameraTiltDirection}
      />

      {/* Touch joysticks — rendered over video when in touch mode */}
      {mode === "touch" && (
        <NippleJoysticks onLeftMove={setTouchLeft} onRightMove={setTouchRight} />
      )}

      <ControlSchemeToggle mode={mode} setMode={setMode} gamepadConnected={gamepadConnected} />
      <PluginControls />

      {/* Settings gear — top-right corner */}
      <button
        onClick={() => setSettingsOpen(true)}
        className="absolute top-4 right-4 z-30 p-2 text-white/40 hover:text-white
                   bg-black/30 hover:bg-black/60 rounded-lg transition-colors"
        title="Settings"
      >
        <GearIcon />
      </button>

      {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}

export default App;
