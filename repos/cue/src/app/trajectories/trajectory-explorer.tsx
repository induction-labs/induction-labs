/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
"use client";

import { useState } from "react";
import { api } from "~/trpc/react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { TrajectoryDataDisplay } from "./trajectory-data-display";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Loader2, FileText } from "lucide-react";

export function TrajectoryExplorer() {
  const [filePath, setFilePath] = useState("gs://induction-labs/evals/step_620/step_620/2025-08-05T04-58-25/osworld_eval_x1qilhb3/samples.jsonl");
  const [currentFilePath, setCurrentFilePath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: trajectoryData, isLoading, error: queryError } = api.trajectory.getTrajectoryData.useQuery(
    { filePath: currentFilePath! },
    { enabled: !!currentFilePath }
  );

  const handleLoadData = async () => {
    if (!filePath.trim()) {
      setError("Please enter a valid GCS file path");
      return;
    }

    setError(null);
    setCurrentFilePath(filePath.trim());
  };

  const handleFilePathChange = (value: string) => {
    setFilePath(value);
    // Clear previous data and errors when changing file path
    if (error) {
      setError(null);
    }
  };

  // Use query error if it exists
  const displayError = error ?? queryError?.message;


  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <FileText className="mr-2 h-5 w-5" />
            Load Trajectory Data
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="file-path">GCS File Path</Label>
            <Input
              id="file-path"
              placeholder="gs://bucket-name/path/to/file.jsonl"
              value={filePath}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleFilePathChange(e.target.value)}
              className="font-mono text-sm"
            />
            <p className="text-sm text-muted-foreground">
              Enter the full Google Cloud Storage path to your JSONL trajectory file
            </p>
          </div>

          <Button
            onClick={handleLoadData}
            disabled={isLoading ?? !filePath.trim()}
            className="w-full sm:w-auto"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading...
              </>
            ) : (
              "Load Trajectory Data"
            )}
          </Button>
        </CardContent>
      </Card>

      {displayError && (
        <Alert variant="destructive">
          <AlertDescription>{displayError}</AlertDescription>
        </Alert>
      )}

      {trajectoryData && <TrajectoryDataDisplay trajectoryData={trajectoryData} />}
    </div>
  );
}