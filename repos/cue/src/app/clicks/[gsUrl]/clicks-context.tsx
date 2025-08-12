"use client";

import { createContext, useContext, type ReactNode } from "react";
import { type ClickEvalRecord } from "~/lib/schemas/clicks";

interface ClicksContextType {
  data?: {
    records: ClickEvalRecord[];
    totalCount: number;
    filePath: string;
  };
  error?: string;
}

const ClicksContext = createContext<ClicksContextType | null>(null);

interface ClicksContextProviderProps {
  children: ReactNode;
  data?: {
    records: ClickEvalRecord[];
    totalCount: number;
    filePath: string;
  };
  error?: string;
}

export function ClicksContextProvider({ children, data, error }: ClicksContextProviderProps) {
  return (
    <ClicksContext.Provider value={{ data, error }}>
      {children}
    </ClicksContext.Provider>
  );
}

export function useClicksData() {
  const context = useContext(ClicksContext);
  if (!context) {
    throw new Error("useClicksData must be used within ClicksContextProvider");
  }
  return context;
}