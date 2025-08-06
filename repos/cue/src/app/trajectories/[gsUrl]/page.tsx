"use client";

import { TrajectoryDataDisplay } from "../trajectory-data-display";
import { Button } from "~/components/ui/button";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useTrajectoryContext } from "./trajectory-context";

export default function TrajectoryDetailPage() {
  const { trajectoryData, gsUrl } = useTrajectoryContext();
  const decodedPath = decodeURIComponent(gsUrl);

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4">
        <div className="mb-8">
          <Link href="/trajectories">
            <Button variant="outline" className="mb-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Trajectory Explorer
            </Button>
          </Link>
          <h1 className="text-4xl font-bold tracking-tight">Trajectory Data</h1>
          <p className="text-muted-foreground mt-2">
            Viewing data from: <code className="bg-muted px-1 py-0.5 rounded text-xs">{decodedPath}</code>
          </p>
        </div>

        {trajectoryData && (
          <TrajectoryDataDisplay trajectoryData={trajectoryData} gsUrl={gsUrl} />
        )}
      </div>
    </main>
  );
}