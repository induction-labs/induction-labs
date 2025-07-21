import { Badge } from "~/components/ui/badge";
import { type JobStatus } from "~/lib/types/kueue";

interface JobStatusBadgeProps {
  status: JobStatus;
}

export function JobStatusBadge({ status }: JobStatusBadgeProps) {
  const getStatusVariant = (status: JobStatus) => {
    switch (status) {
      case 'Running':
        return 'default';
      case 'Admitted':
        return 'secondary';
      case 'Pending':
        return 'outline';
      case 'Suspended':
        return 'secondary';
      case 'Finished':
        return 'default';
      case 'Failed':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  return (
    <Badge variant={getStatusVariant(status)}>
      {status}
    </Badge>
  );
}