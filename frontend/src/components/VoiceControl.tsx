import { useVoiceCommands } from "../hooks/useVoiceCommands";
import { usePlugins } from "../hooks/usePlugins";
import type { SpeedTier } from "../hooks/useControls";

interface Props {
  onTakeoff: () => void;
  onLand: () => void;
  onSpeed: (tier: SpeedTier) => void;
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
      className="w-3.5 h-3.5 flex-shrink-0"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="17" x2="12" y2="21" />
      <line x1="8" y1="21" x2="16" y2="21" />
    </svg>
  );
}

export function VoiceControl({ onTakeoff, onLand, onSpeed }: Props) {
  const { pluginsEnabled, availablePlugins, runningPlugins, togglePlugin } =
    usePlugins();

  const { listening, supported, toggle, lastCommand } = useVoiceCommands({
    onTakeoff,
    onLand,
    onSpeed: (t) => onSpeed(t as SpeedTier),
    onPlugin:
      !pluginsEnabled ? undefined : (
        (name) => {
          if (!availablePlugins.includes(name)) return;
          const wasRunning = runningPlugins.has(name);
          togglePlugin(name).then(() => {
            window.dispatchEvent(
              new CustomEvent(wasRunning ? "plugin:stopped" : "plugin:running"),
            );
          });
        }
      ),
  });

  if (!supported) return null;

  return (
    <>
      {/* Pill button — sits to the left of the gear icon in App.tsx's top-right cluster */}
      <button
        onClick={toggle}
        title='Di "drone land", "drone takeoff", "drone follow", "drone speed high"…'
        className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-full border text-[0.6rem]
                    font-mono tracking-widest uppercase transition-all duration-200 select-none
                    ${
                      listening ?
                        "bg-danger/20 border-danger text-danger shadow-[0_0_12px_rgba(239,83,80,0.4)] animate-pulse"
                      : "bg-black/60 border-white/30 text-white/70 hover:border-white/60 hover:text-white"
                    }`}
      >
        <MicIcon />
        {listening ? "on" : ""}
      </button>

      {/* Command toast — appears below the top-right cluster */}
      {lastCommand && (
        <div
          className="fixed top-20 right-4 z-30 px-3 py-1.5 rounded-lg
                        text-[0.65rem] font-mono tracking-wider select-none pointer-events-none
                        bg-black/80 border border-accent/50 text-accent"
        >
          ✓ {lastCommand}
        </div>
      )}
    </>
  );
}
