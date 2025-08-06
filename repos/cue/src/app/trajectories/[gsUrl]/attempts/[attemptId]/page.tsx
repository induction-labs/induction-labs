"use client";

import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { ArrowLeft, Play, Eye } from "lucide-react";
import Link from "next/link";
import { Badge } from "~/components/ui/badge";
import { useTrajectoryContext } from "../../trajectory-context";
import { useMemo, useState, useEffect } from "react";
import { api } from "~/trpc/react";
import { TrajectoryStepViewer } from "./trajectory-step-viewer";

interface TrajectoryViewerPageProps {
  params: Promise<{ gsUrl: string; attemptId: string }>;
}

export default function TrajectoryViewerPage({ params }: TrajectoryViewerPageProps) {
  const { trajectoryData, gsUrl } = useTrajectoryContext();
  const [attemptId, setAttemptId] = useState<string | null>(null);
  
  useEffect(() => {
    params.then((p) => setAttemptId(p.attemptId));
  }, [params]);

  const decodedGsUrl = decodeURIComponent(gsUrl);
  const decodedAttemptId = attemptId ? decodeURIComponent(attemptId) : "";

  // Find the specific trajectory record for this attempt
  const trajectoryRecord = useMemo(() => {
    if (!trajectoryData?.records || !decodedAttemptId) return null;
    return trajectoryData.records.find(record => record.attempt_id === decodedAttemptId);
  }, [trajectoryData, decodedAttemptId]);

  // Fetch trajectory steps data
  const { data: trajectorySteps, isLoading: stepsLoading, error: stepsError } = api.trajectory.getTrajectorySteps.useQuery(
    { gsUrl, attemptId: decodedAttemptId },
    { enabled: !!gsUrl && !!decodedAttemptId }
  );

  if (!attemptId) {
    return <div>Loading...</div>;
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4">
        <div className="mb-8">
          <Link href={`/trajectories/${gsUrl}`}>
            <Button variant="outline" className="mb-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Trajectory Data
            </Button>
          </Link>
          <h1 className="text-4xl font-bold tracking-tight">Trajectory Viewer</h1>
          <div className="text-muted-foreground mt-2 space-y-1">
            <p>
              Dataset: <code className="bg-muted px-1 py-0.5 rounded text-xs">{decodedGsUrl}</code>
            </p>
            <p>
              Attempt ID: <code className="bg-muted px-1 py-0.5 rounded text-xs">{decodedAttemptId}</code>
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Trajectory Metadata Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Eye className="mr-2 h-5 w-5" />
                Trajectory Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Status</p>
                  <Badge variant={typeof trajectoryRecord?.reward === "string" ? "destructive" : "default"}>
                    {typeof trajectoryRecord?.reward === "string" ? "Failed" : "Completed"}
                  </Badge>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Steps</p>
                  <p className="text-2xl font-bold">{trajectoryRecord?.trajectory_length ?? "-"}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Final Reward</p>
                  <p className="text-2xl font-bold">
                    {trajectoryRecord?.reward !== undefined 
                      ? (typeof trajectoryRecord.reward === "string" 
                          ? trajectoryRecord.reward 
                          : trajectoryRecord.reward.toFixed(3))
                      : "-"
                    }
                  </p>
                </div>
              </div>
              {trajectoryRecord && (
                <div className="mt-6 space-y-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-2">Task ID</p>
                    <code className="bg-muted px-2 py-1 rounded text-sm">{trajectoryRecord.eval_task_id}</code>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-2">Instruction</p>
                    <p className="text-sm bg-muted p-3 rounded">{trajectoryRecord.instruction}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Trajectory Steps */}
          {stepsLoading ? (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Play className="mr-2 h-5 w-5" />
                  Trajectory Steps
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <p>Loading trajectory steps...</p>
                </div>
              </CardContent>
            </Card>
          ) : stepsError ? (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Play className="mr-2 h-5 w-5" />
                  Trajectory Steps
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-destructive">
                  <p>Failed to load trajectory steps: {stepsError.message}</p>
                </div>
              </CardContent>
            </Card>
          ) : trajectorySteps ? (
            <TrajectoryStepViewer steps={trajectorySteps} />
          ) : null}

          {/* Additional Analysis Card */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis & Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <p>Trajectory analysis and insights will be displayed here</p>
                <p className="text-sm mt-2">This could include performance metrics, decision points, etc.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}