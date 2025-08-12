"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { FileText } from "lucide-react";

export function ClicksExplorer() {
  const router = useRouter();
  const [filePath, setFilePath] = useState("gs://induction-labs/evals/clicks/eval_results.jsonl");
  const [error, setError] = useState<string | null>(null);

  const handleLoadData = async () => {
    if (!filePath.trim()) {
      setError("Please enter a valid GCS file path");
      return;
    }

    setError(null);
    const encodedPath = encodeURIComponent(filePath.trim());
    router.push(`/clicks/${encodedPath}`);
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
      <div className="space-y-4">
        <div className="flex items-center mb-4">
          <FileText className="mr-2 h-5 w-5" />
          <span className="font-medium">Load Evaluation Results</span>
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="file-path">GCS File Path</Label>
          <Input
            id="file-path"
            placeholder="gs://bucket-name/path/to/eval_results.jsonl"
            value={filePath}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleFilePathChange(e.target.value)}
            className="font-mono text-sm"
          />
          <p className="text-sm text-muted-foreground">
            Enter the full Google Cloud Storage path to your JSONL evaluation results file
          </p>
        </div>

        <Button
          onClick={handleLoadData}
          disabled={!filePath.trim()}
          className="w-full sm:w-auto"
        >
          Load Evaluation Data
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}