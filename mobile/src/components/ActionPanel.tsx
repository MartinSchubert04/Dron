import type { Capabilities, SpeedTier } from '../hooks/useDrone';

interface Props {
  open:        boolean;
  capabilities: Capabilities;
  speedTier:   SpeedTier;
  onTakeoff:   () => void;
  onLand:      () => void;
  onEstop:     () => void;
  onSpeed:     (t: SpeedTier) => void;
  onTiltStart: (dir: -1 | 1) => void;
  onTiltEnd:   () => void;
}

const CMD_BASE = 'flex-1 py-3 rounded-lg text-sm font-bold tracking-widest transition-all active:scale-95 border';

export function ActionPanel({
  open, capabilities, speedTier,
  onTakeoff, onLand, onEstop,
  onSpeed, onTiltStart, onTiltEnd,
}: Props) {
  return (
    /* slide-down from below the TopBar */
    <div
      className={`absolute inset-x-0 top-14 z-20 transition-all duration-200 ease-out
                  ${open ? 'opacity-100 translate-y-0 pointer-events-auto'
                         : 'opacity-0 -translate-y-2 pointer-events-none'}`}
    >
      <div className="mx-3 bg-surface/90 backdrop-blur-md border border-frame/80
                      rounded-xl shadow-2xl p-4 flex flex-col gap-4">

        {/* ── Commands ─────────────────────────────────── */}
        <div className="flex gap-2">
          {capabilities.takeoff && (
            <button onPointerDown={onTakeoff}
              className={`${CMD_BASE} bg-success/20 border-success/60 text-success`}>
              TAKEOFF
            </button>
          )}
          {capabilities.land && (
            <button onPointerDown={onLand}
              className={`${CMD_BASE} bg-warn/20 border-warn/60 text-warn`}>
              LAND
            </button>
          )}
          {capabilities.estop && (
            <button onPointerDown={onEstop}
              className={`${CMD_BASE} bg-danger/20 border-danger/60 text-danger`}>
              E-STOP
            </button>
          )}
        </div>

        {/* ── Speed ────────────────────────────────────── */}
        {capabilities.speed_control && (
          <div className="flex items-center gap-3">
            <span className="text-muted text-[0.6rem] tracking-widest uppercase w-12">Speed</span>
            <div className="flex flex-1 rounded-lg overflow-hidden border border-frame/60">
              {([0, 1, 2] as SpeedTier[]).map((t, i) => (
                <button
                  key={t}
                  onPointerDown={() => onSpeed(t)}
                  className={`flex-1 py-2 text-xs font-bold tracking-widest transition-colors
                    ${i > 0 ? 'border-l border-frame/60' : ''}
                    ${speedTier === t
                      ? 'bg-accent text-black'
                      : 'bg-surface text-muted hover:text-white'}`}
                >
                  {['LOW', 'MED', 'HIGH'][t]}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Camera tilt ──────────────────────────────── */}
        {capabilities.camera_tilt && (
          <div className="flex items-center gap-3">
            <span className="text-muted text-[0.6rem] tracking-widest uppercase w-12">Cam</span>
            <div className="flex flex-1 gap-2">
              <button
                onPointerDown={() => onTiltStart(1)}
                onPointerUp={onTiltEnd}
                onPointerLeave={onTiltEnd}
                className={`${CMD_BASE} bg-frame/60 border-frame/80 text-white`}
              >
                ▲ UP
              </button>
              <button
                onPointerDown={() => onTiltStart(-1)}
                onPointerUp={onTiltEnd}
                onPointerLeave={onTiltEnd}
                className={`${CMD_BASE} bg-frame/60 border-frame/80 text-white`}
              >
                ▼ DOWN
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
