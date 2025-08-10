import { TrajectoryExplorer } from "./trajectory-explorer";
import { HydrateClient } from "~/trpc/server";

export default async function TrajectoriesPage() {
  return (
    <HydrateClient>
      <main className="min-h-screen bg-background">
        <div className="container mx-auto py-8 px-4">
          <div className="mb-8">
            <h1 className="text-4xl font-bold tracking-tight">Data Trajectory Explorer</h1>
            <p className="text-muted-foreground mt-2">
              Explore and analyze trajectory data from evaluation tasks
            </p>
          </div>
          
          <TrajectoryExplorer />
        </div>
      </main>
    </HydrateClient>
  );
}