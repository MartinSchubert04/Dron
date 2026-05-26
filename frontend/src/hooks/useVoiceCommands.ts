import { useState, useEffect, useRef, useCallback } from "react";

export interface VoiceActions {
  onTakeoff?: () => void;
  onLand?: () => void;
  onSpeed?: (tier: 0 | 1 | 2) => void;
  onPlugin?: (name: string) => void;
}

const WAKE = /\bdrone\b/i;

const RULES: { match: RegExp; fn: (a: VoiceActions) => void; label: string }[] =
  [
    {
      match: /\b(take\s?off|despegar)\b/i,
      fn: (a) => a.onTakeoff?.(),
      label: "Takeoff",
    },
    {
      match: /\b(land|aterrizar|descend)\b/i,
      fn: (a) => a.onLand?.(),
      label: "Land",
    },
    {
      match: /\b(follow|seguir)\b/i,
      fn: (a) => a.onPlugin?.("follow"),
      label: "Follow",
    },
    {
      match: /\b(flip|voltear)\b/i,
      fn: (a) => a.onPlugin?.("flip"),
      label: "Flip",
    },
    {
      match: /\b(speed\s+)?(low|slow|lento|bajo)\b/i,
      fn: (a) => a.onSpeed?.(0),
      label: "Speed → Low",
    },
    {
      match: /\b(speed\s+)?(med(ium)?|medio)\b/i,
      fn: (a) => a.onSpeed?.(1),
      label: "Speed → Med",
    },
    {
      match: /\b(speed\s+)?(high|fast|rápido)\b/i,
      fn: (a) => a.onSpeed?.(2),
      label: "Speed → High",
    },
  ];

export function useVoiceCommands(actions: VoiceActions) {
  const [listening, setListening] = useState(false);
  const [lastCommand, setLastCommand] = useState<string | null>(null);
  const [supported, setSupported] = useState(true);

  const actionsRef = useRef(actions);
  const listeningRef = useRef(false);
  const recRef = useRef<any>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    actionsRef.current = actions;
  });

  const flash = useCallback((label: string) => {
    setLastCommand(label);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setLastCommand(null), 2500);
  }, []);

  useEffect(() => {
    const SR =
      (window as any).SpeechRecognition ??
      (window as any).webkitSpeechRecognition;
    if (!SR) {
      console.warn("[voice] SpeechRecognition not supported");
      setSupported(false);
      return;
    }

    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = false;
    rec.lang = "en-US";
    recRef.current = rec;

    rec.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (!e.results[i].isFinal) continue;
        const t = e.results[i][0].transcript.trim();
        if (!WAKE.test(t)) continue;
        for (const rule of RULES) {
          if (rule.match.test(t)) {
            console.log(`[voice] recognized: "${t}" → ${rule.label}`);
            rule.fn(actionsRef.current);
            flash(rule.label);
            break;
          }
        }
      }
    };

    rec.onend = () => {
      console.log(
        "[voice] recognition ended, restarting:",
        listeningRef.current,
      );
      if (listeningRef.current) {
        try {
          rec.start();
        } catch {
          /* ignore */
        }
      } else {
        setListening(false);
      }
    };

    rec.onerror = (e: any) => {
      console.warn("[voice] error:", e.error);
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        setSupported(false);
        listeningRef.current = false;
        setListening(false);
      }
    };

    return () => {
      listeningRef.current = false;
      rec.onend = null;
      rec.abort();
    };
  }, [flash]);

  const toggle = useCallback(() => {
    const rec = recRef.current;
    if (!rec || !supported) return;
    if (listeningRef.current) {
      console.log("[voice] stopping");
      listeningRef.current = false;
      rec.stop();
      setListening(false);
    } else {
      console.log("[voice] starting");
      listeningRef.current = true;
      setListening(true);
      try {
        rec.start();
      } catch {
        /* already running */
      }
    }
  }, [supported]);

  return { listening, supported, toggle, lastCommand };
}
