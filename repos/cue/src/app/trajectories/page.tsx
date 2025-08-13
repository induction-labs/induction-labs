import Link from "next/link";
import { TrajectoryExplorer } from "./trajectory-explorer";
import { HydrateClient } from "~/trpc/server";

export default async function TrajectoriesPage() {
  return (
    <HydrateClient>
      <main className="min-h-screen bg-background">
        <div className="container mx-auto py-8 px-4">
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-bold tracking-tight">Data Trajectory Explorer</h1>
                <p className="text-muted-foreground mt-2">
                  Explore and analyze trajectory data from evaluation tasks
                </p>
              </div>
              <div className="flex space-x-4">
                <Link href="/clicks" className="text-sm text-muted-foreground hover:text-foreground">
                  Clicks →
                </Link>
                <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
                  Queue Viewer
                </Link>
              </div>
            </div>
          </div>
          
          <TrajectoryExplorer />
        </div>
      </main>
    </HydrateClient>
  );
}