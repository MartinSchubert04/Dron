import { useEffect, useRef, useState } from 'react';

interface Props {
  onLeft:  (throttle: number, yaw:  number) => void;
  onRight: (pitch:    number, roll: number) => void;
}

const BASE = 150;
const KNOB = 52;
const MAX  = (BASE - KNOB) / 2;

function clampToCircle(dx: number, dy: number) {
  const d = Math.sqrt(dx * dx + dy * dy);
  return d > MAX ? { x: dx * MAX / d, y: dy * MAX / d } : { x: dx, y: dy };
}

function JoyZone({ label, sub, onMove, onEnd }: {
  label: string;
  sub: string;
  onMove: (x: number, y: number) => void;
  onEnd: () => void;
}) {
  const [pos,  setPos]  = useState({ x: 0, y: 0 });
  const [snap, setSnap] = useState(false);
  const pid    = useRef<number | null>(null);
  const origin = useRef({ x: 0, y: 0 });

  return (
    <div className="flex flex-col items-center gap-2 select-none">

      <div className="flex flex-col items-center gap-0.5">
        <span className="text-[0.55rem] tracking-[4px] font-mono text-accent/60 uppercase">{label}</span>
        <span className="text-[0.42rem] tracking-widest text-muted/70">{sub}</span>
      </div>

      <div
        className="relative rounded-full touch-none"
        style={{
          width: BASE,
          height: BASE,
          background: '#0f0f1a',
          border: '2px solid #2a2a3a',
          boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.5)',
        }}
        onPointerDown={e => {
          if (pid.current !== null) return;
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          pid.current = e.pointerId;
          origin.current = { x: e.clientX, y: e.clientY };
          setSnap(false);
        }}
        onPointerMove={e => {
          if (e.pointerId !== pid.current) return;
          const p = clampToCircle(
            e.clientX - origin.current.x,
            e.clientY - origin.current.y,
          );
          setPos(p);
          onMove(p.x / MAX, -(p.y / MAX));
        }}
        onPointerUp={e => {
          if (e.pointerId !== pid.current) return;
          pid.current = null;
          setSnap(true);
          setPos({ x: 0, y: 0 });
          onEnd();
        }}
        onPointerCancel={e => {
          if (e.pointerId !== pid.current) return;
          pid.current = null;
          setSnap(true);
          setPos({ x: 0, y: 0 });
          onEnd();
        }}
      >
        {/* Crosshair */}
        <div className="absolute inset-0 rounded-full overflow-hidden pointer-events-none">
          <div className="absolute top-1/2 -translate-y-px"
               style={{ left: 12, right: 12, height: 1, background: '#252535' }} />
          <div className="absolute left-1/2 -translate-x-px"
               style={{ top: 12, bottom: 12, width: 1, background: '#252535' }} />
        </div>

        {/* Knob */}
        <div
          className="absolute rounded-full pointer-events-none"
          style={{
            width:      KNOB,
            height:     KNOB,
            left:       '50%',
            top:        '50%',
            marginLeft: -KNOB / 2,
            marginTop:  -KNOB / 2,
            background: 'radial-gradient(circle at 38% 35%, #7dd9fc, #4fc3f7)',
            border:     '2px solid rgba(79,195,247,0.55)',
            boxShadow:  '0 0 14px rgba(79,195,247,0.7), 0 0 5px rgba(79,195,247,0.4)',
            transform:  `translate(${pos.x}px, ${pos.y}px)`,
            transition: snap ? 'transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)' : 'none',
          }}
        />
      </div>
    </div>
  );
}

export function NippleJoysticks({ onLeft, onRight }: Props) {
  const leftCb  = useRef(onLeft);
  const rightCb = useRef(onRight);
  useEffect(() => { leftCb.current  = onLeft;  }, [onLeft]);
  useEffect(() => { rightCb.current = onRight; }, [onRight]);

  return (
    <>
      <div className="absolute bottom-0 left-0 z-10 p-5"
           style={{ paddingBottom: 'max(20px, env(safe-area-inset-bottom))' }}>
        <JoyZone
          label="THR / YAW"
          sub="▲▼ Throttle  ·  ◀▶ Yaw"
          onMove={(x, y) => leftCb.current(y, x)}
          onEnd={() => leftCb.current(0, 0)}
        />
      </div>

      <div className="absolute bottom-0 right-0 z-10 p-5"
           style={{ paddingBottom: 'max(20px, env(safe-area-inset-bottom))' }}>
        <JoyZone
          label="PITCH / ROLL"
          sub="▲▼ Pitch  ·  ◀▶ Roll"
          onMove={(x, y) => rightCb.current(y, x)}
          onEnd={() => rightCb.current(0, 0)}
        />
      </div>
    </>
  );
}
