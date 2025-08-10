import { z } from "zod";

// Schema for a single trajectory record from the JSONL file
export const trajectoryRecordSchema = z.object({
  eval_task_id: z.string(),
  attempt_id: z.string(),
  instruction: z.string(),
  output_folder: z.string(),
  trajectory_length: z.number(),
  reward: z.union([z.number(), z.string()]),
});

// Schema for the complete trajectory data response
export const trajectoryDataSchema = z.object({
  records: z.array(trajectoryRecordSchema),
  totalCount: z.number(),
  filePath: z.string(),
});

// Schema for GCS file path input
export const gcsFilePathSchema = z.object({
  filePath: z.string().regex(/^gs:\/\//, "File path must start with 'gs://'"),
});

// Schema for individual trajectory step
export const trajectoryStepSchema = z.object({
  step: z.number(),
  image: z.string(),
  action: z.string(),
  text: z.string(),
});

// Schema for trajectory steps data
export const trajectoryStepsSchema = z.array(trajectoryStepSchema);

// Schema for trajectory steps request
export const trajectoryStepsRequestSchema = z.object({
  gsUrl: z.string(),
  attemptId: z.string(),
});

// Type exports
export type TrajectoryRecord = z.infer<typeof trajectoryRecordSchema>;
export type TrajectoryData = z.infer<typeof trajectoryDataSchema>;
export type GCSFilePath = z.infer<typeof gcsFilePathSchema>;
export type TrajectoryStep = z.infer<typeof trajectoryStepSchema>;
export type TrajectorySteps = z.infer<typeof trajectoryStepsSchema>;