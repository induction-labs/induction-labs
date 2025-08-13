import { z } from "zod";

// Schema for individual click evaluation record
export const clickEvalRecordSchema = z.object({
  id: z.string(),
  instruction: z.string(),
  image_path: z.string(),
  gt_x1: z.number().int(),
  gt_x2: z.number().int(),
  gt_y1: z.number().int(),
  gt_y2: z.number().int(),
  pred_x: z.number().int().nullable(),
  pred_y: z.number().int().nullable(),
  is_in_bbox: z.boolean(),
  latency_seconds: z.number(),
  raw_response: z.string(),
  center_coords: z.tuple([z.number().int(), z.number().int()]).nullable(),
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