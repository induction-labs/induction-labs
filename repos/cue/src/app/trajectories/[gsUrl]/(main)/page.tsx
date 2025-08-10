"use client";

import { TrajectoryDataDisplay } from "../../trajectory-data-display";
import { useTrajectoryContext } from "../trajectory-context";

export default function TrajectoryMainPage() {
  const { trajectoryData, gsUrl } = useTrajectoryContext();

  return trajectoryData ? (
    <TrajectoryDataDisplay trajectoryData={trajectoryData} gsUrl={gsUrl} />
  ) : null;
}