import { useState, useEffect, useRef } from 'react';
import { useSettings } from '../context/SettingsContext';

export interface OverlayObject {
  type: 'rect';
  coords: [number, number, number, number]; // [x1, y1, x2, y2]
  color: string;
}

export function useOverlays() {
  const { wsBase } = useSettings();
  const [overlays, setOverlays] = useState<OverlayObject[]>([]);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(`${wsBase}/ws/overlays`);

      ws.current.onopen = () => console.log('%c[Overlays] WebSocket Connected', 'color: green');
      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (Array.isArray(data)) setOverlays(data);
        } catch (e) {
          console.error('[Overlays] Failed to parse data:', e);
        }
      };

      ws.current.onclose = () => {
        console.log('%c[Overlays] WebSocket Disconnected. Reconnecting...', 'color: orange');
        setTimeout(connect, 3000);
      };

      ws.current.onerror = (err) => {
        console.error('%c[Overlays] WebSocket Error', 'color: red', err);
        ws.current?.close();
      };
    };

    connect();

    return () => {
      if (ws.current) {
        ws.current.onclose = null; // prevent the reconnect timer from firing
        ws.current.close();
      }
    };
  }, [wsBase]); // reconnect when host changes

  return overlays;
}
