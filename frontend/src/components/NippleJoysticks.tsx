import { useEffect, useRef } from 'react';
import nipplejs from 'nipplejs';

interface Props {
  onLeftMove:  (throttle: number, yaw:  number) => void; // -1..1 each
  onRightMove: (pitch:    number, roll: number) => void; // -1..1 each
}

const ZONE_SIZE = 160;

export function NippleJoysticks({ onLeftMove, onRightMove }: Props) {
  const leftRef  = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);

  // Keep callbacks stable so nipplejs init only runs once
  const leftCb  = useRef(onLeftMove);
  const rightCb = useRef(onRightMove);
  useEffect(() => { leftCb.current  = onLeftMove; }, [onLeftMove]);
  useEffect(() => { rightCb.current = onRightMove; }, [onRightMove]);

  useEffect(() => {
    if (!leftRef.current || !rightRef.current) return;

    const left = nipplejs.create({
      zone:        leftRef.current,
      mode:        'static',
      position:    { left: '50%', top: '50%' },
      color:       '#4fc3f7',
      size:        ZONE_SIZE * 0.7,
      restOpacity: 0.35,
    });

    const right = nipplejs.create({
      zone:        rightRef.current,
      mode:        'static',
      position:    { left: '50%', top: '50%' },
      color:       '#4fc3f7',
      size:        ZONE_SIZE * 0.7,
      restOpacity: 0.35,
    });

    // Left stick: throttle (up = +1) + yaw (right = +1)
    left.on('move', (_, d) => leftCb.current(d.vector.y, d.vector.x));
    left.on('end',  ()     => leftCb.current(0, 0));

    // Right stick: pitch (up = +1 = forward) + roll (right = +1)
    right.on('move', (_, d) => rightCb.current(d.vector.y, d.vector.x));
    right.on('end',  ()     => rightCb.current(0, 0));

    return () => {
      // Zero axes on unmount so drone doesn't freeze mid-input
      leftCb.current(0, 0);
      rightCb.current(0, 0);
      left.destroy();
      right.destroy();
    };
  }, []);

  const zoneClass = `joy-zone relative bg-surface/60 border border-frame/80 rounded-2xl backdrop-blur-sm`;

  return (
    <div className="absolute inset-x-0 bottom-0 h-[55%] z-20 flex items-end pb-6 px-6
                    gap-4 pointer-events-none">

      {/* Left — Throttle / Yaw */}
      <div className="pointer-events-auto flex flex-col items-center gap-1.5 flex-1">
        <span className="text-accent text-[0.55rem] tracking-[3px] select-none">THR / YAW</span>
        <div
          ref={leftRef}
          className={zoneClass}
          style={{ width: ZONE_SIZE, height: ZONE_SIZE }}
        />
        <span className="text-muted text-[0.45rem] select-none">▲ Subir · ◀▶ Girar</span>
      </div>

      {/* Center spacer — leaves room for command buttons */}
      <div className="flex-[2]" />

      {/* Right — Pitch / Roll */}
      <div className="pointer-events-auto flex flex-col items-center gap-1.5 flex-1">
        <span className="text-accent text-[0.55rem] tracking-[3px] select-none">PITCH / ROLL</span>
        <div
          ref={rightRef}
          className={zoneClass}
          style={{ width: ZONE_SIZE, height: ZONE_SIZE }}
        />
        <span className="text-muted text-[0.45rem] select-none">▲▼ Adelante / Atrás · ◀▶ Ladear</span>
      </div>
    </div>
  );
}
