import { z } from "zod";



export const k8sWorkloadOwnerFields = z.object({
  controller: z.literal(true),
  kind: z.literal("Job"),
  name: z.string(),
  uid: z.string(),
})

// Basic Kubernetes metadata schema
export const k8sMetadataSchema = z.object({
  name: z.string(),
  namespace: z.string(),
  uid: z.string(),
  creationTimestamp: z.string(),
  labels: z.record(z.string()).optional(),
  annotations: z.record(z.string()).optional(),
  ownerReferences: z.array(k8sWorkloadOwnerFields).length(1),

});

// Condition schema for status conditions
export const conditionSchema = z.object({
  type: z.string(),
  status: z.string(),
  lastTransitionTime: z.string(),
  reason: z.string().optional(),
  message: z.string().optional(),
});

// Admission check schema
export const admissionCheckSchema = z.object({
  name: z.string(),
  state: z.string(),
  message: z.string().optional(),
  lastTransitionTime: z.string().optional(),
});

// Container schema for job template
export const containerSchema = z.object({
  name: z.string(),
  image: z.string(),
  command: z.array(z.string()).optional(),
  args: z.array(z.string()).optional(),
});

// Job template schema
export const jobTemplateSchema = z.object({
  spec: z.object({
    template: z.object({
      spec: z.object({
        containers: z.array(containerSchema),
      }),
    }),
  }),
});

// Kueue job spec schema
export const kueueJobSpecSchema = z.object({
  queueName: z.string(),
  priority: z.number().optional(),
  suspend: z.boolean().optional(),
  // jobTemplate: jobTemplateSchema,
});

// Kueue job status schema
export const kueueJobStatusSchema = z.object({
  conditions: z.array(conditionSchema).optional(),
  admissionChecks: z.array(admissionCheckSchema).optional(),
}).optional();

// Full Kueue job schema
export const kueueJobSchema = z.object({
  metadata: k8sMetadataSchema,
  spec: kueueJobSpecSchema,
  status: kueueJobStatusSchema,
});

// Schema for Kubernetes API list response
export const kueueJobListSchema = z.object({
  items: z.array(kueueJobSchema),
  metadata: z.object({
    resourceVersion: z.string().optional(),
    continue: z.string().optional(),
  }).optional(),
});

// export const k8sApiResponse = <T extends z.ZodTypeAny>(schema: T) => z.object({
//   response: schema
// });
// Job status enum schema
export const jobStatusSchema = z.enum(['Pending', 'Admitted', 'Running', 'Suspended', 'Finished', 'Failed']);

// Schema for our internal job list item
export const jobListItemSchema = z.object({
  name: z.string(),
  namespace: z.string(),
  queueName: z.string(),
  status: jobStatusSchema,
  priority: z.number().optional(),
  createdAt: z.string(),
  admittedAt: z.string().optional(),
  suspended: z.boolean(),
});

// Type exports
export type KueueJob = z.infer<typeof kueueJobSchema>;
export type KueueJobList = z.infer<typeof kueueJobListSchema>;
export type JobStatus = z.infer<typeof jobStatusSchema>;
export type JobListItem = z.infer<typeof jobListItemSchema>;