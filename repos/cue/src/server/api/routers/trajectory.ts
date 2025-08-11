import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { gcsClient } from "~/lib/gcs";
import { gcsFilePathSchema, trajectoryStepsRequestSchema, trajectoryStepsSchema, trajectoryMetadataRequestSchema, trajectoryMetadataSchema } from "~/lib/schemas/trajectory";

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

  getTrajectorySteps: publicProcedure
    .input(trajectoryStepsRequestSchema)
    .query(async ({ input }) => {
      try {
        // Convert gsUrl from *.jsonl to metadata/{attempt_id}.json
        const decodedGsUrl = decodeURIComponent(input.gsUrl);

        // Extract the prefix by removing '/*.jsonl' from the end
        const prefix = decodedGsUrl.replace(/\/[^/]+\.jsonl$/, '');

        // Construct the metadata path
        const metadataPath = `${prefix}/metadata/${input.attemptId}.json`;

        // Read and validate the JSON file (now just trajectory steps)
        const stepsData = await gcsClient.readJSONFile(metadataPath);
        const validatedSteps = trajectoryStepsSchema.parse(stepsData);
        
        return validatedSteps;
      } catch (error) {
        console.error('Error in getTrajectorySteps:', error);
        throw new Error(error instanceof Error ? error.message : 'Failed to fetch trajectory steps');
      }
    }),

  getTrajectoryMetadata: publicProcedure
    .input(trajectoryMetadataRequestSchema)
    .query(async ({ input }) => {
      try {
        // Convert gsUrl from *.jsonl to train_samples/{attempt_id}.metadata.json
        const decodedGsUrl = decodeURIComponent(input.gsUrl);

        // Extract the prefix by removing '/*.jsonl' from the end
        const prefix = decodedGsUrl.replace(/\/[^/]+\.jsonl$/, '');

        // Construct the metadata path
        const metadataPath = `${prefix}/train_samples/${input.attemptId}.metadata.json`;

        // Read and validate the JSON file
        const metadataData = await gcsClient.readJSONFile(metadataPath);
        const validatedMetadata = trajectoryMetadataSchema.parse(metadataData);
        
        return validatedMetadata;
      } catch (error) {
        console.warn('Trajectory metadata not found:', error instanceof Error ? error.message : 'Unknown error');
        // Return null instead of throwing error when metadata is not found
        return null;
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