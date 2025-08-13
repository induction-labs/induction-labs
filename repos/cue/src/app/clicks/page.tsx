import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { ClicksExplorer } from "./clicks-explorer";

export default function ClicksPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4">
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold tracking-tight">Clicks Data Explorer</h1>
              <p className="text-muted-foreground mt-2">
                Load and explore evaluation results from Google Cloud Storage
              </p>
            </div>
            <div className="flex space-x-4">
              <Link href="/trajectories" className="text-sm text-muted-foreground hover:text-foreground">
                ← Trajectories
              </Link>
              <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
                Queue Viewer
              </Link>
            </div>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Load Evaluation Data</CardTitle>
          </CardHeader>
          <CardContent>
            <ClicksExplorer />
          </CardContent>
        </Card>
      </div>
    </main>
  );
}