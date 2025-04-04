
import React, { createContext, useContext, useEffect, useState } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleHighContrast: () => void;
  isHighContrast: boolean;
  increaseFontSize: () => void;
  decreaseFontSize: () => void;
  fontSizeLevel: number;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  const [isHighContrast, setIsHighContrast] = useState(() => {
    return localStorage.getItem("highContrast") === "true";
  });

  const [fontSizeLevel, setFontSizeLevel] = useState(() => {
    const stored = parseInt(localStorage.getItem("fontSizeLevel") || "0");
    return isNaN(stored) ? 0 : stored;
  });

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    const root = window.document.documentElement;
    if (isHighContrast) {
      root.classList.add("high-contrast");
    } else {
      root.classList.remove("high-contrast");
    }
    localStorage.setItem("highContrast", isHighContrast.toString());
  }, [isHighContrast]);

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("larger-text", "extra-large-text");
    
    if (fontSizeLevel === 1) {
      root.classList.add("larger-text");
    } else if (fontSizeLevel === 2) {
      root.classList.add("extra-large-text");
    }
    
    localStorage.setItem("fontSizeLevel", fontSizeLevel.toString());
  }, [fontSizeLevel]);

  const toggleHighContrast = () => {
    setIsHighContrast(prev => !prev);
  };

  const increaseFontSize = () => {
    setFontSizeLevel(prev => Math.min(prev + 1, 2));
  };

  const decreaseFontSize = () => {
    setFontSizeLevel(prev => Math.max(prev - 1, 0));
  };

  return (
    <ThemeContext.Provider value={{ 
      theme, 
      setTheme, 
      toggleHighContrast, 
      isHighContrast,
      increaseFontSize,
      decreaseFontSize,
      fontSizeLevel
    }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
