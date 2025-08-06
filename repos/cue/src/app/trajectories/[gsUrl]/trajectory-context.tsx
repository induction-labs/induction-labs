"use client";

import { createContext, useContext, type ReactNode } from "react";
import { type TrajectoryData } from "~/lib/schemas/trajectory";

interface TrajectoryContextType {
  trajectoryData: TrajectoryData | undefined;
  isLoading: boolean;
  error: string | undefined;
  gsUrl: string;
}

const TrajectoryContext = createContext<TrajectoryContextType | undefined>(undefined);

interface TrajectoryProviderProps {
  children: ReactNode;
  trajectoryData: TrajectoryData | undefined;
  isLoading: boolean;
  error: string | undefined;
  gsUrl: string;
}

export function TrajectoryProvider({ 
  children, 
  trajectoryData, 
  isLoading, 
  error, 
  gsUrl 
}: TrajectoryProviderProps) {
  return (
    <TrajectoryContext.Provider value={{ trajectoryData, isLoading, error, gsUrl }}>
      {children}
    </TrajectoryContext.Provider>
  );
}

export function useTrajectoryContext() {
  const context = useContext(TrajectoryContext);
  if (context === undefined) {
    throw new Error("useTrajectoryContext must be used within a TrajectoryProvider");
  }
  return context;
}