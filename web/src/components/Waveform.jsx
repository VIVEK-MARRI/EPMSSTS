import { useEffect, useRef } from "react";

export default function Waveform({ buffer }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !buffer) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
    ctx.strokeStyle = "rgba(139, 92, 246, 0.8)";
    ctx.lineWidth = 1.5;

    const raw = buffer.getChannelData(0);
    const step = Math.ceil(raw.length / width);
    const amp = height / 2;

    ctx.beginPath();
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = raw[i * step + j] || 0;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }
    ctx.stroke();
  }, [buffer]);

  return (
    <canvas
      ref={canvasRef}
      width={420}
      height={120}
      className="w-full rounded-xl bg-base-700/60 border border-white/10"
    />
  );
}
