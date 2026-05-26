import { useRef } from 'react';

interface Props { src: string }

export function VideoFeed({ src }: Props) {
  const ref = useRef<HTMLImageElement>(null);
  return (
    <img
      ref={ref}
      src={src}
      alt=""
      className="absolute inset-0 w-full h-full object-cover"
      onError={() => { if (ref.current) ref.current.style.visibility = 'hidden'; }}
      onLoad={() =>  { if (ref.current) ref.current.style.visibility = 'visible'; }}
    />
  );
}
