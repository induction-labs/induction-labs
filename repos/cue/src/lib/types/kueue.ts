// Re-export types from schemas for backward compatibility
export type {
  KueueJob,
  KueueJobList,
  JobStatus,
  JobListItem,
} from "~/lib/schemas/kueue";

// Additional utility types for workload status (not directly from Kueue API)
export interface WorkloadStatus {
  admitted: boolean;
  conditions?: Array<{
    type: string;
    status: string;
    lastTransitionTime: string;
    reason?: string;
    message?: string;
  }>;
  queuedAt?: string;
  admittedAt?: string;
}