export function VideoFeed({ src, flipped = false }: { src: string; flipped?: boolean }) {
  return (
    <img
      src={src}
      alt="Drone video feed"
      className="absolute inset-0 w-full h-full object-contain transition-transform duration-300"
      style={flipped ? { transform: "rotate(180deg)" } : undefined}
    />
  );
}
