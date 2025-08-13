"use client";

import Link from "next/link";
import { Button } from "~/components/ui/button";
import { ChevronLeft } from "lucide-react";
import { ClicksDataDisplay } from "./clicks-data-display";
import { useClicksData } from "./clicks-context";
import { EditablePathInput } from "~/components/editable-path-input";

interface ClicksDataPageContentProps {
  gsUrl: string;
  decodedPath: string;
}

export function ClicksDataPageContent({ gsUrl, decodedPath }: ClicksDataPageContentProps) {
  const { data: clicksData, error } = useClicksData();

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <Button variant="outline" size="sm" asChild>
              <Link href="/clicks">
                <ChevronLeft className="mr-2 h-4 w-4" />
                Back to Clicks
              </Link>
            </Button>
          </div>
          <div>
            <h1 className="text-4xl font-bold tracking-tight">Evaluation Results</h1>
            <div className="mt-2">
              <p className="text-muted-foreground text-sm mb-2">Viewing data from:</p>
              <EditablePathInput 
                currentPath={decodedPath}
                basePath="/clicks"
              />
            </div>
          </div>
        </div>

        {/* Content */}
        {error ? (
          <div className="text-center py-12">
            <p className="text-destructive mb-4">{error}</p>
            <Button variant="outline" asChild>
              <Link href="/clicks">Try Another File</Link>
            </Button>
          </div>
        ) : clicksData ? (
          <ClicksDataDisplay data={clicksData} gsUrl={gsUrl} />
        ) : (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Loading evaluation data...</p>
          </div>
        )}
      </div>
    </main>
  );
}