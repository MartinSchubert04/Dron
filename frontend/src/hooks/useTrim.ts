import { useCallback, useEffect, useState } from "react";
import { useSettings } from "../context/SettingsContext";

export interface TrimValues {
  roll:     number;
  pitch:    number;
  yaw:      number;
  throttle: number;
}

const ZERO: TrimValues = { roll: 0, pitch: 0, yaw: 0, throttle: 0 };

export function useTrim() {
  const { apiBase } = useSettings();
  const [trim, setTrimState]   = useState<TrimValues>(ZERO);
  const [saving, setSaving]    = useState(false);
  const [saveMsg, setSaveMsg]  = useState("");

  useEffect(() => {
    fetch(`${apiBase}/trim`)
      .then(r => r.json())
      .then(j => setTrimState({ roll: j.roll ?? 0, pitch: j.pitch ?? 0, yaw: j.yaw ?? 0, throttle: j.throttle ?? 0 }))
      .catch(() => {});
  }, [apiBase]);

  const save = useCallback(async (values: TrimValues) => {
    setSaving(true);
    setSaveMsg("");
    try {
      const params = new URLSearchParams({
        roll:     String(values.roll),
        pitch:    String(values.pitch),
        yaw:      String(values.yaw),
        throttle: String(values.throttle),
      });
      const res = await fetch(`${apiBase}/trim?${params}`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setTrimState(values);
      setSaveMsg("Guardado ✓");
    } catch (e) {
      setSaveMsg(`Error: ${e}`);
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(""), 3000);
    }
  }, [apiBase]);

  return { trim, setTrimState, save, saving, saveMsg };
}
