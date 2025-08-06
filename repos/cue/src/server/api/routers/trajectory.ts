import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { gcsClient } from "~/lib/gcs";
import { gcsFilePathSchema } from "~/lib/schemas/trajectory";

export const trajectoryRouter = createTRPCRouter({
  getTrajectoryData: publicProcedure
    .input(gcsFilePathSchema)
    .query(async ({ input }) => {
      try {
        const records = await gcsClient.readJSONLFile(input.filePath);
        
        return {
          records,
          totalCount: records.length,
          filePath: input.filePath,
        };
      } catch (error) {
        console.error('Error in getTrajectoryData:', error);
        throw new Error(error instanceof Error ? error.message : 'Failed to fetch trajectory data');
      }
    }),

  listFiles: publicProcedure
    .input(z.object({ 
      bucketName: z.string(),
      prefix: z.string().optional() 
    }))
    .query(async ({ input }) => {
      try {
        return await gcsClient.listFiles(input.bucketName, input.prefix);
      } catch (error) {
        console.error(`Error listing files in bucket ${input.bucketName}:`, error);
        throw new Error(error instanceof Error ? error.message : `Failed to list files in bucket ${input.bucketName}`);
      }
    }),
});