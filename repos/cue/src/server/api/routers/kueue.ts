import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { type JobListItem } from "~/lib/types/kueue";

const mockJobs: JobListItem[] = [
  {
    name: "training-job-1",
    namespace: "ml-team",
    queueName: "gpu-queue",
    status: "Running",
    priority: 10,
    createdAt: "2024-01-15T10:30:00Z",
    admittedAt: "2024-01-15T10:35:00Z",
    suspended: false,
  },
  {
    name: "data-processing-job",
    namespace: "analytics",
    queueName: "cpu-queue",
    status: "Pending",
    priority: 5,
    createdAt: "2024-01-15T11:00:00Z",
    suspended: false,
  },
  {
    name: "model-inference-job",
    namespace: "ml-team",
    queueName: "gpu-queue",
    status: "Admitted",
    priority: 8,
    createdAt: "2024-01-15T09:45:00Z",
    admittedAt: "2024-01-15T09:50:00Z",
    suspended: false,
  },
  {
    name: "batch-processing-job",
    namespace: "data-team",
    queueName: "cpu-queue",
    status: "Suspended",
    priority: 3,
    createdAt: "2024-01-15T08:20:00Z",
    suspended: true,
  },
  {
    name: "training-job-2",
    namespace: "ml-team",
    queueName: "gpu-queue",
    status: "Failed",
    priority: 7,
    createdAt: "2024-01-15T07:10:00Z",
    admittedAt: "2024-01-15T07:15:00Z",
    suspended: false,
  },
  {
    name: "etl-pipeline",
    namespace: "data-team",
    queueName: "memory-queue",
    status: "Finished",
    priority: 6,
    createdAt: "2024-01-15T06:00:00Z",
    admittedAt: "2024-01-15T06:05:00Z",
    suspended: false,
  },
];

export const kueueRouter = createTRPCRouter({
  getJobs: publicProcedure
    .query(async () => {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      return mockJobs;
    }),

  getJobsByNamespace: publicProcedure
    .input(z.object({ namespace: z.string() }))
    .query(async ({ input }) => {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 300));
      return mockJobs.filter(job => job.namespace === input.namespace);
    }),
});