import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { kubernetesClient } from "~/lib/kubernetes";

export const kueueRouter = createTRPCRouter({
  getJobs: publicProcedure
    .query(async () => {
      try {
        return await kubernetesClient.getKueueJobs();
      } catch (error) {
        console.error('Error in getJobs:', error);
        throw new Error(error instanceof Error ? error.message : 'Failed to fetch Kueue jobs');
      }
    }),

  getJobsByNamespace: publicProcedure
    .input(z.object({ namespace: z.string() }))
    .query(async ({ input }) => {
      try {
        return await kubernetesClient.getKueueJobsByNamespace(input.namespace);
      } catch (error) {
        console.error(`Error in getJobsByNamespace for ${input.namespace}:`, error);
        throw new Error(error instanceof Error ? error.message : `Failed to fetch jobs for namespace ${input.namespace}`);
      }
    }),

  getJob: publicProcedure
    .input(z.object({ 
      namespace: z.string(), 
      name: z.string() 
    }))
    .query(async ({ input }) => {
      try {
        return await kubernetesClient.getKueueJob(input.namespace, input.name);
      } catch (error) {
        console.error(`Error in getJob for ${input.namespace}/${input.name}:`, error);
        throw new Error(error instanceof Error ? error.message : `Failed to fetch job ${input.namespace}/${input.name}`);
      }
    }),
});