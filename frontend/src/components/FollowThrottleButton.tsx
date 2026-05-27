import { usePlugins } from '../hooks/usePlugins';

const PLUGIN_NAME = 'FollowThrottlePlugin';

export function FollowThrottleButton() {
  const { pluginsEnabled, availablePlugins, runningPlugins, togglePlugin } = usePlugins();

  if (!pluginsEnabled || !availablePlugins.includes(PLUGIN_NAME)) {
    return null;
  }

  const isRunning = runningPlugins.has(PLUGIN_NAME);

  return (
    <button
      onClick={async () => {
        await togglePlugin(PLUGIN_NAME);
        window.dispatchEvent(new CustomEvent(isRunning ? 'plugin:stopped' : 'plugin:running'));
      }}
      title="Follow (with throttle hold)"
      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors
        ${isRunning
          ? 'bg-cyan-500 hover:bg-cyan-600 text-black'
          : 'bg-black/40 hover:bg-black/60 text-white/80 border border-white/20'
        }`}
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
           strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
        <circle cx="12" cy="8" r="4" />
        <path d="M6 20v-2a6 6 0 0 1 12 0v2" />
        <path d="M19 12l2 2-2 2" />
        <path d="M5 12l-2 2 2 2" />
      </svg>
      Follow+
    </button>
  );
}
