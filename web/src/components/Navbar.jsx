import { motion } from "framer-motion";

export default function Navbar({ online, onToggleTheme, theme }) {
  return (
    <motion.nav
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full px-6 py-5 flex items-center justify-between border-b border-white/10 bg-base-900/80 backdrop-blur sticky top-0 z-50"
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-500 to-aura-600 flex items-center justify-center font-bold text-lg">
          E
        </div>
        <div>
          <h1 className="text-xl font-semibold tracking-tight">EPMSSTS</h1>
          <p className="text-xs text-white/60">Emotion-Preserving Speech AI</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm text-white/70 bg-white/5 px-3 py-2 rounded-full border border-white/10">
          <span className={`dot-pulse ${online ? "bg-emerald-400" : "bg-rose-400"}`} />
          <span className="text-xs">{online ? "Online" : "Offline"}</span>
        </div>
      </div>
    </motion.nav>
  );
}
