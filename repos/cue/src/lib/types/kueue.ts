export interface KueueJob {
  metadata: {
    name: string;
    namespace: string;
    uid: string;
    creationTimestamp: string;
    labels?: Record<string, string>;
    annotations?: Record<string, string>;
  };
  spec: {
    queueName: string;
    priority?: number;
    suspend?: boolean;
    jobTemplate: {
      spec: {
        template: {
          spec: {
            containers: Array<{
              name: string;
              image: string;
              command?: string[];
              args?: string[];
            }>;
          };
        };
      };
    };
  };
  status?: {
    conditions?: Array<{
      type: string;
      status: string;
      lastTransitionTime: string;
      reason?: string;
      message?: string;
    }>;
    admissionChecks?: Array<{
      name: string;
      state: string;
      message?: string;
      lastTransitionTime?: string;
    }>;
  };
}

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

export type JobStatus = 'Pending' | 'Admitted' | 'Running' | 'Suspended' | 'Finished' | 'Failed';

export interface JobListItem {
  name: string;
  namespace: string;
  queueName: string;
  status: JobStatus;
  priority?: number;
  createdAt: string;
  admittedAt?: string;
  suspended: boolean;
}