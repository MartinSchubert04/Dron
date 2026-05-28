import { createContext, useCallback, useContext, useMemo, useState } from 'react';

interface Settings {
  backendHost: string;
  esp32Url: string;
}

interface SettingsContextValue {
  settings: Settings;
  updateSettings: (partial: Partial<Settings>) => void;
  apiBase: string;
  wsBase: string;
}

const DEFAULT_SETTINGS: Settings = {
  backendHost: 'localhost:8000',
  esp32Url: '',
};

const LS_KEY = 'turbodrone_settings';

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch { /* ignore corrupt data */ }
  return DEFAULT_SETTINGS;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(loadSettings);

  const updateSettings = useCallback((partial: Partial<Settings>) => {
    setSettings(prev => {
      const next = { ...prev, ...partial };
      localStorage.setItem(LS_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const value = useMemo<SettingsContextValue>(() => ({
    settings,
    updateSettings,
    apiBase: `http://${settings.backendHost}`,
    wsBase:  `ws://${settings.backendHost}`,
  }), [settings, updateSettings]);

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used inside <SettingsProvider>');
  return ctx;
}
