/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,jsx}"] ,
  theme: {
    extend: {
      colors: {
        base: {
          900: "#0B1020",
          800: "#11182B",
          700: "#151E36",
          600: "#1C2744"
        },
        accent: {
          500: "#38BDF8",
          600: "#0EA5E9"
        },
        aura: {
          500: "#8B5CF6",
          600: "#7C3AED"
        },
        emotion: {
          happy: "#22C55E",
          sad: "#3B82F6",
          angry: "#EF4444",
          neutral: "#9CA3AF",
          fearful: "#F59E0B"
        }
      },
      boxShadow: {
        glow: "0 0 30px rgba(56, 189, 248, 0.25)",
        soft: "0 10px 25px rgba(0, 0, 0, 0.25)"
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New", "monospace"]
      }
    }
  },
  plugins: []
};
