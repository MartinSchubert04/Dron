import { useCallback, useState } from 'react';

interface Settings {
  backendHost: string;
}

const DEFAULT: Settings = { backendHost: 'localhost:8000' };
const LS_KEY = 'turbodrone_mobile_settings';

function load(): Settings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) return { ...DEFAULT, ...JSON.parse(raw) };
  } catch { /* ignore */ }
  return DEFAULT;
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings>(load);

  const update = useCallback((partial: Partial<Settings>) => {
    setSettings(prev => {
      const next = { ...prev, ...partial };
      localStorage.setItem(LS_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  return {
    settings,
    update,
    apiBase: `http://${settings.backendHost}`,
    wsBase:  `ws://${settings.backendHost}`,
  };
}
