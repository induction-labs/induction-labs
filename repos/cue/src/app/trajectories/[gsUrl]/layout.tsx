import { type ReactNode } from "react";
import { api } from "~/trpc/server";
import { TrajectoryProvider } from "./trajectory-context";
import { Alert, AlertDescription } from "~/components/ui/alert";

interface LayoutProps {
  children: ReactNode;
  params: Promise<{ gsUrl: string }>;
}

export default async function TrajectoryLayout({ children, params }: LayoutProps) {
  const { gsUrl } = await params;
  const decodedPath = decodeURIComponent(gsUrl);

  let trajectoryData;
  let error;

  try {
    trajectoryData = await api.trajectory.getTrajectoryData({ filePath: decodedPath });
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load trajectory data";
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <TrajectoryProvider 
      trajectoryData={trajectoryData}
      isLoading={false}
      error={error}
      gsUrl={gsUrl}
    >
      {children}
    </TrajectoryProvider>
  );
}