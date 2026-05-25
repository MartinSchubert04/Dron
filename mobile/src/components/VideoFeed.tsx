interface Props { src: string }

export function VideoFeed({ src }: Props) {
  return (
    <img
      src={src}
      alt="drone feed"
      className="absolute inset-0 w-full h-full object-cover"
    />
  );
}
