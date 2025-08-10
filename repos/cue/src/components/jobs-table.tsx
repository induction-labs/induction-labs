"use client";

import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { JobStatusBadge } from "~/components/job-status-badge";
import { type JobListItem } from "~/lib/types/kueue";

interface JobsTableProps {
  jobs: JobListItem[];
}

export function JobsTable({ jobs }: JobsTableProps) {
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (createdAt: string, admittedAt?: string) => {
    if (!admittedAt) return '-';

    const created = new Date(createdAt);
    const admitted = new Date(admittedAt);
    const diffMs = admitted.getTime() - created.getTime();
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffMinutes < 60) {
      return `${diffMinutes}m`;
    } else {
      const hours = Math.floor(diffMinutes / 60);
      const minutes = diffMinutes % 60;
      return `${hours}h ${minutes}m`;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queue Jobs ({jobs.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Namespace</TableHead>
                <TableHead>Queue</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Priority</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Wait Time</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                    No jobs found
                  </TableCell>
                </TableRow>
              ) : (
                jobs.map((job) => (
                  <TableRow key={`${job.namespace}/${job.name}`} className="hover:bg-muted/50">
                    <TableCell className="font-medium">
                      <Link
                        href={`/jobs/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.name)}`}
                        className="block w-full hover:underline"
                      >
                        {job.name}
                      </Link>
                    </TableCell>
                    <TableCell className="text-muted-foreground">{job.namespace}</TableCell>
                    <TableCell>{job.queueName}</TableCell>
                    <TableCell>
                      <JobStatusBadge status={job.status} />
                    </TableCell>
                    <TableCell>{job.priority ?? '-'}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatTimestamp(job.createdAt)}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDuration(job.createdAt, job.admittedAt)}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}