import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { gcsClient } from "~/lib/gcs";
import { clicksGcsFilePathSchema } from "~/lib/schemas/clicks";

export const clicksRouter = createTRPCRouter({
  getClickEvalData: publicProcedure
    .input(clicksGcsFilePathSchema)
    .query(async ({ input }) => {
      try {
        const records = await gcsClient.readClickEvalJSONLFile(input.filePath);

        return {
          records,
          totalCount: records.length,
          filePath: input.filePath,
        };
      } catch (error) {
        console.error('Error in getClickEvalData:', error);
        throw new Error(error instanceof Error ? error.message : 'Failed to fetch click evaluation data');
      }
    }),
});