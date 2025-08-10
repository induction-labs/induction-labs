"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { FileText } from "lucide-react";

export function TrajectoryExplorer() {
  const router = useRouter();
  const [filePath, setFilePath] = useState("gs://induction-labs/evals/step_620/step_620/2025-08-05T04-58-25/osworld_eval_x1qilhb3/samples.jsonl");
  const [error, setError] = useState<string | null>(null);

  const handleLoadData = async () => {
    if (!filePath.trim()) {
      setError("Please enter a valid GCS file path");
      return;
    }

    setError(null);
    const encodedPath = encodeURIComponent(filePath.trim());
    router.push(`/trajectories/${encodedPath}`);
  };

  const handleFilePathChange = (value: string) => {
    setFilePath(value);
    // Clear previous errors when changing file path
    if (error) {
      setError(null);
    }
  };


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
            disabled={!filePath.trim()}
            className="w-full sm:w-auto"
          >
            Load Trajectory Data
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}