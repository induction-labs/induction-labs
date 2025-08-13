"use client";

import { type ReactNode } from "react";
import { Button } from "~/components/ui/button";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useTrajectoryContext } from "../trajectory-context";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "~/components/ui/tabs";
import { usePathname } from "next/navigation";
import { EditablePathInput } from "~/components/editable-path-input";

interface MainLayoutProps {
  children: ReactNode;
}

export default function MainLayout({ children }: MainLayoutProps) {
  const { gsUrl } = useTrajectoryContext();
  const decodedPath = decodeURIComponent(gsUrl);
  const pathname = usePathname();
  
  // Determine active tab based on pathname
  const activeTab = pathname.endsWith('/stats') ? 'stats' : 'overview';

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
          <div className="mt-2">
            <p className="text-muted-foreground text-sm mb-2">Viewing data from:</p>
            <EditablePathInput 
              currentPath={decodedPath}
              basePath="/trajectories"
            />
          </div>
        </div>

        <Tabs value={activeTab} className="space-y-6">
          <TabsList>
            <TabsTrigger value="overview" asChild>
              <Link href={`/trajectories/${gsUrl}`}>Overview</Link>
            </TabsTrigger>
            <TabsTrigger value="stats" asChild>
              <Link href={`/trajectories/${gsUrl}/stats`}>Statistics</Link>
            </TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab}>
            {children}
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}