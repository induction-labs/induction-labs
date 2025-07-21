"use client";

import { api } from "~/trpc/react";
import { JobsTable } from "~/components/jobs-table";

export function QueueViewer() {
  const [jobs] = api.kueue.getJobs.useSuspenseQuery();

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold tracking-tight">Kueue Queue Viewer</h1>
          <p className="text-muted-foreground mt-2">
            Monitor and manage jobs in your Kubernetes cluster queues
          </p>
        </div>

        <JobsTable jobs={jobs} />
      </div>
    </main>
  );
}