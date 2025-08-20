import { z } from "zod";

// Schema for individual click evaluation record

export const clickInputSchema = z.object({
  id: z.string(),
  image_url: z.string().url(),
  instruction: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  x1: z.number(),
  y1: z.number(),
  x2: z.number(),
  y2: z.number(),
});


export const clickEvalRecordSchema = z.object({
  input: clickInputSchema,
  response: z.object({
    raw_response: z.string(),
  }).passthrough(),
  prompt_text: z.string(),
  prediction_point: z.tuple([z.number(), z.number()]).nullable(),
  center_coords: z.tuple([z.number(), z.number()]),
  is_in_bbox: z.boolean(),
  x_error: z.number().nullable(),
  y_error: z.number().nullable(),
  pixel_distance: z.number().nullable(),
});

export type ClickEvalRecord = z.infer<typeof clickEvalRecordSchema>;

// Schema for the full evaluation data response
export const clickEvalDataSchema = z.object({
  records: z.array(clickEvalRecordSchema),
  totalCount: z.number(),
  filePath: z.string(),
});

export type ClickEvalData = z.infer<typeof clickEvalDataSchema>;

// Schema for GCS file path request
export const clicksGcsFilePathSchema = z.object({
  filePath: z.string().min(1, "File path is required"),
});