import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
} from 'react';
import { DEBUG, DEBUG_IP } from './config';
import {
  Animated,
  Alert,
  PanResponder,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

// ── Theme ─────────────────────────────────────────────────────────────────────

const C = {
  bg:      '#0d0d0d',
  panel:   '#1a1a1a',
  border:  '#2a2a2a',
  text:    '#ffffff',
  muted:   '#666',
  armed:   '#e74c3c',
  safe:    '#2ecc71',
  accent:  '#3498db',
};

// ── Joystick ──────────────────────────────────────────────────────────────────

interface JoystickProps {
  label: string;
  onMove: (x: number, y: number) => void;
}

function Joystick({ label, onMove }: JoystickProps) {
  const BASE = 160;
  const KNOB = 54;
  const MAX  = (BASE - KNOB) / 2;   // max px offset from center

  const pos = useRef(new Animated.ValueXY()).current;

  const pan = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder:  () => true,
      onPanResponderMove: (_, g) => {
        const x = Math.max(-MAX, Math.min(MAX, g.dx));
        const y = Math.max(-MAX, Math.min(MAX, g.dy));
        pos.setValue({ x, y });
        onMove(x / MAX, -(y / MAX));   // y inverted: up = +1
      },
      onPanResponderRelease: () => {
        Animated.spring(pos, {
          toValue: { x: 0, y: 0 },
          useNativeDriver: true,
          friction: 5,
          tension: 80,
        }).start();
        onMove(0, 0);
      },
    })
  ).current;

  return (
    <View style={styles.joystickWrap}>
      <Text style={styles.joystickLabel}>{label}</Text>
      <View style={[styles.joystickBase, { width: BASE, height: BASE, borderRadius: BASE / 2 }]}>
        {/* crosshair lines */}
        <View style={[styles.crossH, { width: BASE - 20 }]} />
        <View style={[styles.crossV, { height: BASE - 20 }]} />
        <Animated.View
          {...pan.panHandlers}
          style={[
            styles.joystickKnob,
            { width: KNOB, height: KNOB, borderRadius: KNOB / 2 },
            { transform: [{ translateX: pos.x }, { translateY: pos.y }] },
          ]}
        />
      </View>
    </View>
  );
}

// ── Motor bar ─────────────────────────────────────────────────────────────────

function MotorBar({ label, value }: { label: string; value: number }) {
  const pct = value / 255;
  const color = pct > 0.7 ? C.armed : pct > 0.3 ? '#f39c12' : C.accent;
  return (
    <View style={styles.motorBox}>
      <Text style={styles.motorLabel}>{label}</Text>
      <View style={styles.motorTrack}>
        <View style={[styles.motorFill, { height: `${pct * 100}%`, backgroundColor: color }]} />
      </View>
      <Text style={styles.motorVal}>{value}</Text>
    </View>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

type Screen = 'connect' | 'control';

export default function App() {
  const [screen,    setScreen]  = useState<Screen>(DEBUG ? 'control' : 'connect');
  const [ip,        setIp]      = useState(DEBUG ? DEBUG_IP : '192.168.1.');
  const [status,    setStatus]  = useState('Desconectado');
  const [armed,     setArmed]   = useState(false);
  const [motors,    setMotors]  = useState([0, 0, 0, 0]);

  const wsRef    = useRef<WebSocket | null>(null);
  const leftJoy  = useRef({ x: 0, y: 0 });   // x = yaw,  y = throttle
  const rightJoy = useRef({ x: 0, y: 0 });   // x = roll, y = pitch
  const armedRef = useRef(false);
  const cmdTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── WebSocket send ──────────────────────────────────────────────────────────

  const send = useCallback((obj: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }, []);

  // ── Command loop (50 ms) ────────────────────────────────────────────────────

  const stopLoop = useCallback(() => {
    if (cmdTimer.current) { clearInterval(cmdTimer.current); cmdTimer.current = null; }
  }, []);

  const startLoop = useCallback(() => {
    stopLoop();
    cmdTimer.current = setInterval(() => {
      if (!armedRef.current) return;
      const t = Math.round(Math.max(0, leftJoy.current.y)  * 255);
      const y = Math.round(leftJoy.current.x  * 127);
      const p = Math.round(rightJoy.current.y * 127);
      const r = Math.round(rightJoy.current.x * 127);
      send({ cmd: 'move', t, y, p, r });
    }, 50);
  }, [send, stopLoop]);

  // ── Connect / Disconnect ────────────────────────────────────────────────────

  const connect = useCallback(() => {
    const addr = ip.trim();
    if (!addr) return;
    setStatus('Conectando…');

    const socket = new WebSocket(`ws://${addr}:81`);
    wsRef.current = socket;

    socket.onopen = () => {
      setStatus(`ws://${addr}:81`);
      setScreen('control');
      startLoop();
    };
    socket.onclose = () => {
      setStatus('Desconectado');
      setArmed(false);
      armedRef.current = false;
      setScreen('connect');
      stopLoop();
    };
    socket.onerror = () => {
      setStatus('Error');
      Alert.alert('Error', `No se pudo conectar a ws://${addr}:81`);
    };
    socket.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data as string);
        if (Array.isArray(d.motors)) setMotors(d.motors);
      } catch { /* ignore */ }
    };
  }, [ip, startLoop, stopLoop]);

  const disconnect = useCallback(() => {
    send({ cmd: 'disarm' });
    wsRef.current?.close();
  }, [send]);

  // ── Arm / Disarm ────────────────────────────────────────────────────────────

  const toggleArm = useCallback(() => {
    if (!armedRef.current) {
      armedRef.current = true;
      setArmed(true);
      send({ cmd: 'arm' });
    } else {
      armedRef.current = false;
      setArmed(false);
      send({ cmd: 'disarm' });
    }
  }, [send]);

  useEffect(() => () => { stopLoop(); wsRef.current?.close(); }, [stopLoop]);

  // Auto-connect in debug mode
  useEffect(() => { if (DEBUG) connect(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Connect screen ──────────────────────────────────────────────────────────

  if (screen === 'connect') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar style="light" />
        <View style={styles.card}>
          <Text style={styles.title}>Drone Controller</Text>
          <Text style={styles.fieldLabel}>IP del ESP32</Text>
          <TextInput
            style={styles.input}
            value={ip}
            onChangeText={setIp}
            placeholder="192.168.1.100"
            placeholderTextColor={C.muted}
            keyboardType="numeric"
            autoCapitalize="none"
            returnKeyType="done"
            onSubmitEditing={connect}
          />
          <Text style={styles.statusText}>{status}</Text>
          <TouchableOpacity style={styles.connectBtn} onPress={connect} activeOpacity={0.8}>
            <Text style={styles.connectBtnText}>Conectar</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ── Control screen ──────────────────────────────────────────────────────────

  return (
    <SafeAreaView style={[styles.container, { justifyContent: 'flex-start' }]}>
      <StatusBar style="light" />

      {/* Header bar */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Drone Controller</Text>
        <View style={[styles.statusBadge, { backgroundColor: armed ? C.armed : '#333' }]}>
          <Text style={styles.statusBadgeText}>{armed ? 'ARMADO' : 'DESARMADO'}</Text>
        </View>
        <Text style={styles.ipText}>{status}</Text>
        <TouchableOpacity onPress={disconnect}>
          <Text style={styles.disconnectText}>Desconectar</Text>
        </TouchableOpacity>
      </View>

      {/* Motor bars */}
      <View style={styles.motorRow}>
        <MotorBar label="FL" value={motors[0]} />
        <MotorBar label="FR" value={motors[1]} />
        <MotorBar label="BL" value={motors[2]} />
        <MotorBar label="BR" value={motors[3]} />
      </View>

      {/* Main controls: left joystick | arm btn | right joystick */}
      <View style={styles.controls}>
        <Joystick
          label="Throttle / Yaw"
          onMove={(x, y) => { leftJoy.current = { x, y }; }}
        />

        <TouchableOpacity
          style={[styles.armBtn, armed && styles.armBtnArmed]}
          onPress={toggleArm}
          activeOpacity={0.85}
        >
          <Text style={styles.armBtnText}>{armed ? 'DISARM' : 'ARM'}</Text>
        </TouchableOpacity>

        <Joystick
          label="Pitch / Roll"
          onMove={(x, y) => { rightJoy.current = { x, y }; }}
        />
      </View>
    </SafeAreaView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // ── Connect ──────────────────────────────────────────────────────────────────
  card: {
    width: 300,
    backgroundColor: C.panel,
    borderRadius: 16,
    padding: 28,
    alignItems: 'stretch',
    borderWidth: 1,
    borderColor: C.border,
  },
  title: {
    color: C.text,
    fontSize: 22,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  fieldLabel: {
    color: C.muted,
    fontSize: 12,
    marginBottom: 6,
  },
  input: {
    backgroundColor: '#111',
    color: C.text,
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: C.border,
  },
  statusText: {
    color: C.muted,
    fontSize: 11,
    marginBottom: 16,
    textAlign: 'center',
  },
  connectBtn: {
    backgroundColor: C.accent,
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
  },
  connectBtnText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 15,
  },

  // ── Header ────────────────────────────────────────────────────────────────
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    paddingHorizontal: 14,
    paddingVertical: 7,
    backgroundColor: C.panel,
    borderBottomWidth: 1,
    borderBottomColor: C.border,
    gap: 10,
  },
  headerTitle: {
    color: C.text,
    fontWeight: 'bold',
    fontSize: 14,
    flex: 1,
  },
  statusBadge: {
    borderRadius: 4,
    paddingHorizontal: 7,
    paddingVertical: 3,
  },
  statusBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  ipText: {
    color: C.muted,
    fontSize: 11,
  },
  disconnectText: {
    color: C.muted,
    fontSize: 12,
  },

  // ── Motors ────────────────────────────────────────────────────────────────
  motorRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 18,
    paddingVertical: 10,
  },
  motorBox: {
    alignItems: 'center',
    gap: 3,
  },
  motorLabel: {
    color: C.muted,
    fontSize: 10,
    fontWeight: 'bold',
  },
  motorTrack: {
    width: 18,
    height: 52,
    backgroundColor: '#1e1e1e',
    borderRadius: 4,
    overflow: 'hidden',
    justifyContent: 'flex-end',
    borderWidth: 1,
    borderColor: C.border,
  },
  motorFill: {
    width: '100%',
    borderRadius: 3,
  },
  motorVal: {
    color: C.text,
    fontSize: 9,
  },

  // ── Controls layout ────────────────────────────────────────────────────────
  controls: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingHorizontal: 16,
    paddingBottom: 12,
  },

  // ── Joystick ──────────────────────────────────────────────────────────────
  joystickWrap: {
    alignItems: 'center',
    gap: 6,
  },
  joystickLabel: {
    color: C.muted,
    fontSize: 11,
  },
  joystickBase: {
    backgroundColor: '#111',
    borderWidth: 2,
    borderColor: C.border,
    justifyContent: 'center',
    alignItems: 'center',
  },
  crossH: {
    position: 'absolute',
    height: 1,
    backgroundColor: '#2a2a2a',
  },
  crossV: {
    position: 'absolute',
    width: 1,
    backgroundColor: '#2a2a2a',
  },
  joystickKnob: {
    position: 'absolute',
    backgroundColor: C.accent,
    borderWidth: 2,
    borderColor: '#5dade2',
    shadowColor: C.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 8,
    elevation: 6,
  },

  // ── Arm button ─────────────────────────────────────────────────────────────
  armBtn: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: C.safe,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#27ae60',
    shadowColor: C.safe,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 5,
  },
  armBtnArmed: {
    backgroundColor: C.armed,
    borderColor: '#c0392b',
    shadowColor: C.armed,
  },
  armBtnText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
    letterSpacing: 1,
  },
});
