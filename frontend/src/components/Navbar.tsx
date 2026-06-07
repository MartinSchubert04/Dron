import type { ConnState } from "../hooks/useControls";

export type Tab = "control" | "3d" | "config";

interface Props {
  activeTab:   Tab;
  setTab:      (t: Tab) => void;
  connState:   ConnState;
  imuConnected: boolean;
  roll:        number;
  pitch:       number;
}

const CONN_DOT: Record<ConnState, string> = {
  connected:    "bg-success shadow-[0_0_6px_#66bb6a]",
  connecting:   "bg-warn animate-pulse",
  disconnected: "bg-danger",
};

const TABS: { id: Tab; label: string }[] = [
  { id: "control", label: "CONTROL" },
  { id: "3d",      label: "3D / IMU" },
  { id: "config",  label: "CONFIG" },
];

export function Navbar({ activeTab, setTab, connState, imuConnected, roll, pitch }: Props) {
  return (
    <nav className="flex-shrink-0 flex items-center gap-4 px-4 h-10
                    bg-surface border-b border-frame z-50 select-none">

      {/* Brand */}
      <span className="font-heading font-bold text-white tracking-[0.18em] text-sm whitespace-nowrap">
        TURBODRONE
      </span>

      {/* Divider */}
      <span className="w-px h-5 bg-frame flex-shrink-0" />

      {/* Tabs */}
      <div className="flex items-stretch h-full gap-0.5">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`px-4 text-[0.65rem] font-mono tracking-widest transition-all border-b-2
              ${activeTab === id
                ? "border-accent text-accent"
                : "border-transparent text-muted hover:text-white hover:border-frame"}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* IMU pill — only shown when connected */}
      {imuConnected && (
        <div className="flex items-center gap-2 text-[0.6rem] font-mono">
          <span className="text-muted">R</span>
          <span className={Math.abs(roll)  > 10 ? "text-warn" : "text-white"}>
            {roll.toFixed(1)}°
          </span>
          <span className="text-muted ml-1">P</span>
          <span className={Math.abs(pitch) > 10 ? "text-warn" : "text-white"}>
            {pitch.toFixed(1)}°
          </span>
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-success ml-1
                           shadow-[0_0_4px_#66bb6a]" />
        </div>
      )}

      {/* Drone connection status */}
      <div className="flex items-center gap-1.5">
        <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${CONN_DOT[connState]}`} />
        <span className="text-[0.6rem] text-muted hidden sm:block">
          {connState === "connected" ? "Online" : connState === "connecting" ? "Connecting…" : "Offline"}
        </span>
      </div>
    </nav>
  );
}
