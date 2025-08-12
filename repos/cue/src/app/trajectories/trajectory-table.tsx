import Link from "next/link";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { ClickableId } from "./clickable-id";
import { DataTable } from "~/components/data-table";
import { type TrajectoryRecord } from "~/lib/schemas/trajectory";

interface TrajectoryTableProps {
  data: TrajectoryRecord[];
  gsUrl?: string;
}

export function TrajectoryTable({ data, gsUrl }: TrajectoryTableProps) {
  const getRewardColor = (reward: string | number) => {
    if (typeof reward === 'string') {
      return "destructive"; // String rewards (like "Internal Fail") are errors
    }
    if (reward >= 0.8) return "default";
    if (reward >= 0.5) return "secondary";
    return "destructive";
  };

  const formatReward = (reward: string | number) => {
    if (typeof reward === 'string') {
      return reward; // Return the string as-is
    }
    return reward.toFixed(3);
  };

  const columns = [
    {
      key: 'eval_task_id' as keyof TrajectoryRecord,
      label: 'Task ID',
      render: (value: TrajectoryRecord[keyof TrajectoryRecord]) => (
        <ClickableId id={String(value)} />
      ),
    },
    {
      key: 'attempt_id' as keyof TrajectoryRecord,
      label: 'Attempt ID',
      render: (value: TrajectoryRecord[keyof TrajectoryRecord]) => (
        <ClickableId id={String(value)} />
      ),
    },
    {
      key: 'instruction' as keyof TrajectoryRecord,
      label: 'Instruction',
      sortable: false,
      className: 'max-w-xl',
      render: (value: TrajectoryRecord[keyof TrajectoryRecord], record: TrajectoryRecord) => {
        const encodedAttemptId = encodeURIComponent(record.attempt_id);
        const trajectoryHref = gsUrl ? `/trajectories/${gsUrl}/attempts/${encodedAttemptId}` : '#';

        return (
          <Tooltip>
            <TooltipTrigger asChild>
              <Link href={trajectoryHref}>
                <div className="line-clamp-2 cursor-pointer hover:text-primary transition-colors">
                  {String(value)}
                </div>
              </Link>
            </TooltipTrigger>
            <TooltipContent className="max-w-md">
              <p className="whitespace-pre-wrap">{String(value)}</p>
            </TooltipContent>
          </Tooltip>
        );
      },
    },
    {
      key: 'trajectory_length' as keyof TrajectoryRecord,
      label: 'Trajectory Length',
      className: 'text-center',
      render: (value: TrajectoryRecord[keyof TrajectoryRecord]) => (
        <Badge variant="outline">{String(value)}</Badge>
      ),
    },
    {
      key: 'reward' as keyof TrajectoryRecord,
      label: 'Reward',
      className: 'text-center',
      render: (value: TrajectoryRecord[keyof TrajectoryRecord]) => {
        if (!value) return null;
        return (
          <Badge variant={getRewardColor(value)}>
            {formatReward(value)}
          </Badge>
        );
      },
    },
  ];

  return (
    <DataTable
      data={data}
      columns={columns}
      title="Trajectory Records"
      emptyMessage="No trajectory data available"
      getRowKey={(record, index) => `${record.eval_task_id}-${record.attempt_id}-${index}`}
    />
  );
}