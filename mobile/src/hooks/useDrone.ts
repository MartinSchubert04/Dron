import { useCallback, useEffect, useRef, useState } from 'react';

export type ConnState = 'connecting' | 'connected' | 'disconnected';

export interface Capabilities {
  takeoff:       boolean;
  land:          boolean;
  estop:         boolean;
  camera_tilt:   boolean;
  speed_control: boolean;
}

export type SpeedTier = 0 | 1 | 2;

const DEFAULT_CAPS: Capabilities = {
  takeoff: true, land: true, estop: true,
  camera_tilt: false, speed_control: false,
};

const SEND_HZ = 30;

export function useDrone(apiBase: string, wsBase: string) {
  const [connState,    setConnState]    = useState<ConnState>('connecting');
  const [droneType,    setDroneType]    = useState('unknown');
  const [capabilities, setCapabilities] = useState<Capabilities>(DEFAULT_CAPS);
  const [speedTier,    setSpeedTierSt]  = useState<SpeedTier>(2);

  const ws            = useRef<WebSocket | null>(null);
  const throttleRef   = useRef(0);
  const yawRef        = useRef(0);
  const pitchRef      = useRef(0);
  const rollRef        = useRef(0);
  const cameraTiltRef = useRef<-1 | 0 | 1>(0);
  const apiBaseRef    = useRef(apiBase);
  useEffect(() => { apiBaseRef.current = apiBase; }, [apiBase]);

  /* ── WebSocket ─────────────────────────────────────────────────────── */
  useEffect(() => {
    setConnState('connecting');
    const socket = new WebSocket(`${wsBase}/ws`);
    ws.current = socket;
    socket.onopen  = () => setConnState('connected');
    socket.onclose = () => setConnState('disconnected');
    socket.onerror = () => setConnState('disconnected');
    return () => {
      socket.onopen = socket.onclose = socket.onerror = null;
      socket.close();
      ws.current = null;
    };
  }, [wsBase]);

  /* ── 30 Hz axes transmission ───────────────────────────────────────── */
  useEffect(() => {
    const id = setInterval(() => {
      if (ws.current?.readyState !== WebSocket.OPEN) return;
      ws.current.send(JSON.stringify({
        type:                 'axes',
        mode:                 'abs',
        camera_tilt_direction: cameraTiltRef.current,
        throttle: throttleRef.current,
        yaw:      yawRef.current,
        pitch:    pitchRef.current,
        roll:     rollRef.current,
      }));
    }, 1000 / SEND_HZ);
    return () => clearInterval(id);
  }, []);

  /* ── Capabilities ──────────────────────────────────────────────────── */
  useEffect(() => {
    let cancelled = false;
    fetch(`${apiBase}/capabilities`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (cancelled || !d) return;
        setDroneType(d.drone_type ?? 'unknown');
        setCapabilities({
          takeoff:       Boolean(d.commands?.takeoff),
          land:          Boolean(d.commands?.land),
          estop:         Boolean(d.commands?.estop),
          camera_tilt:   Boolean(d.commands?.camera_tilt),
          speed_control: Boolean(d.commands?.speed_control),
        });
      })
      .catch(() => { /* keep defaults */ });
    return () => { cancelled = true; };
  }, [apiBase]);

  /* ── Helpers ───────────────────────────────────────────────────────── */
  const send = useCallback((type: string, payload: Record<string, unknown> = {}) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type, ...payload }));
    }
  }, []);

  /* ── Joystick callbacks (called by NippleJoysticks at ~60fps) ──────── */
  const setLeft  = useCallback((throttle: number, yaw: number) => {
    throttleRef.current = throttle;
    yawRef.current      = yaw;
  }, []);

  const setRight = useCallback((pitch: number, roll: number) => {
    pitchRef.current = pitch;
    rollRef.current  = roll;
  }, []);

  /* ── Commands ──────────────────────────────────────────────────────── */
  const takeoff = useCallback(() => send('takeoff'), [send]);
  const land    = useCallback(() => send('land'),    [send]);
  const estop   = useCallback(() => send('estop'),   [send]);

  const setSpeed = useCallback((tier: SpeedTier) => {
    setSpeedTierSt(tier);
    send('set_speed_index', { speed_index: tier });
  }, [send]);

  const setCameraTilt = useCallback((dir: -1 | 0 | 1) => {
    cameraTiltRef.current = dir;
  }, []);

  return {
    connState, droneType, capabilities,
    speedTier, setSpeed,
    setLeft, setRight, setCameraTilt,
    takeoff, land, estop,
  };
}
