import Link from "next/link";
import { ArrowLeft, Hash, Clock, Layers3 } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Badge } from "~/components/ui/badge";
import { JobStatusBadge } from "~/components/job-status-badge";
import { api, HydrateClient } from "~/trpc/server";

interface JobDetailPageProps {
  params: Promise<{
    namespace: string;
    name: string;
  }>;
}

export default async function JobDetailPage({ params }: JobDetailPageProps) {
  const { namespace, name } = await params;
  const job = await api.kueue.getJob({
    namespace: decodeURIComponent(namespace),
    name: decodeURIComponent(name)
  });

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (createdAt: string, admittedAt?: string) => {
    if (!admittedAt) return 'Not admitted yet';

    const created = new Date(createdAt);
    const admitted = new Date(admittedAt);
    const diffMs = admitted.getTime() - created.getTime();
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffMinutes < 60) {
      return `${diffMinutes} minutes`;
    } else {
      const hours = Math.floor(diffMinutes / 60);
      const minutes = diffMinutes % 60;
      return `${hours} hours ${minutes} minutes`;
    }
  };

  return (
    <HydrateClient>
      <div className="min-h-screen bg-background">
        <div className="container mx-auto py-8 px-4">
          <div className="mb-6">
            <Button variant="ghost" asChild className="mb-4">
              <Link href="/">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Queue
              </Link>
            </Button>

            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-bold tracking-tight">{job.name}</h1>
                <p className="text-muted-foreground mt-2">
                  Job details in namespace <Badge variant="outline">{job.namespace}</Badge>
                </p>
              </div>
              <JobStatusBadge status={job.status} />
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Hash className="mr-2 h-5 w-5" />
                  Basic Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Name:</span>
                  <span className="text-sm">{job.name}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Namespace:</span>
                  <Badge variant="outline">{job.namespace}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Queue:</span>
                  <Badge>{job.queueName}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Priority:</span>
                  <span className="text-sm">{job.priority ?? 'Default'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Suspended:</span>
                  <Badge variant={job.suspended ? "destructive" : "secondary"}>
                    {job.suspended ? 'Yes' : 'No'}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Clock className="mr-2 h-5 w-5" />
                  Timing Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Created:</span>
                  <span className="text-sm">{formatTimestamp(job.createdAt)}</span>
                </div>
                {job.admittedAt && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Admitted:</span>
                    <span className="text-sm">{formatTimestamp(job.admittedAt)}</span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Wait Time:</span>
                  <span className="text-sm">{formatDuration(job.createdAt, job.admittedAt)}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Layers3 className="mr-2 h-5 w-5" />
                Job Status Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h3 className="font-medium">Current Status</h3>
                    <p className="text-sm text-muted-foreground">
                      {job.status === 'Running' && 'Job is currently running'}
                      {job.status === 'Pending' && 'Job is waiting to be admitted to a queue'}
                      {job.status === 'Admitted' && 'Job has been admitted and is starting'}
                      {job.status === 'Suspended' && 'Job execution has been suspended'}
                      {job.status === 'Finished' && 'Job completed successfully'}
                      {job.status === 'Failed' && 'Job execution failed'}
                    </p>
                  </div>
                  <JobStatusBadge status={job.status} />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </HydrateClient>
  );
}