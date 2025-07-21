import * as k8s from '@kubernetes/client-node';
import {
  kueueJobListSchema,
  kueueJobSchema,
  jobStatusSchema,
} from './schemas/kueue';
import type {
  KueueJob,
  JobListItem,
  JobStatus
} from './schemas/kueue';

class KubernetesClient {
  private kc: k8s.KubeConfig;
  private customApi: k8s.CustomObjectsApi;

  constructor() {
    this.kc = new k8s.KubeConfig();
    this.kc.loadFromDefault();
    this.customApi = this.kc.makeApiClient(k8s.CustomObjectsApi);
  }

  async getKueueJobs(): Promise<JobListItem[]> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const response = await this.customApi.listClusterCustomObject({
        group: 'kueue.x-k8s.io',
        version: 'v1beta1',
        plural: 'workloads'
      });

      // Validate the response using Zod schema
      const parsedResponse = kueueJobListSchema.safeParse(response);
      if (!parsedResponse.success) {
        console.error('Validation error:', parsedResponse.error);
        throw new Error('Invalid response format from Kueue API');
      }
      const validatedResponse = parsedResponse.data;
      const workloads = validatedResponse.items;

      return workloads.map(workload => this.transformWorkloadToJobItem(workload));
    } catch (error) {
      console.error('Error fetching Kueue jobs:', error);
      if (error instanceof Error) {
        throw new Error(`Failed to fetch Kueue jobs from cluster: ${error.message}`);
      }
      throw new Error('Failed to fetch Kueue jobs from cluster');
    }
  }

  async getKueueJobsByNamespace(namespace: string): Promise<JobListItem[]> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const response = await this.customApi.listNamespacedCustomObject({
        group: 'kueue.x-k8s.io',
        version: 'v1beta1',
        namespace: namespace,
        plural: 'workloads'
      });

      // Validate the response using Zod schema
      const validatedResponse = kueueJobListSchema.parse(response);
      const workloads = validatedResponse.items;

      return workloads.map(workload => this.transformWorkloadToJobItem(workload));
    } catch (error) {
      console.error(`Error fetching Kueue jobs for namespace ${namespace}:`, error);
      if (error instanceof Error) {
        throw new Error(`Failed to fetch Kueue jobs from namespace ${namespace}: ${error.message}`);
      }
      throw new Error(`Failed to fetch Kueue jobs from namespace ${namespace}`);
    }
  }

  async getKueueJob(namespace: string, name: string): Promise<JobListItem> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const response = await this.customApi.getNamespacedCustomObject({
        group: 'kueue.x-k8s.io',
        version: 'v1beta1',
        namespace: namespace,
        plural: 'workloads',
        name: name
      });

      // Validate the response using Zod schema
      const validatedWorkload = kueueJobSchema.parse(response);

      return this.transformWorkloadToJobItem(validatedWorkload);
    } catch (error) {
      console.error(`Error fetching Kueue job ${namespace}/${name}:`, error);
      if (error instanceof Error) {
        // Check if it's a 404 error (job not found)
        if (error.message.includes('404') || error.message.includes('Not Found')) {
          throw new Error(`Job ${namespace}/${name} not found`);
        }
        throw new Error(`Failed to fetch Kueue job ${namespace}/${name}: ${error.message}`);
      }
      throw new Error(`Failed to fetch Kueue job ${namespace}/${name}`);
    }
  }

  private transformWorkloadToJobItem(workload: KueueJob): JobListItem {
    const status = this.getJobStatus(workload);
    const suspended = workload.spec.suspend ?? false;

    return {
      name: workload.metadata.name,
      namespace: workload.metadata.namespace,
      queueName: workload.spec.queueName,
      status,
      priority: workload.spec.priority,
      createdAt: workload.metadata.creationTimestamp,
      admittedAt: this.getAdmittedTime(workload),
      suspended,
    };
  }

  private getJobStatus(workload: KueueJob): JobStatus {
    let status = 'Pending';

    if (workload.status?.conditions) {
      const conditions = workload.status.conditions;

      // Check for completion conditions
      const finishedCondition = conditions.find(c => c.type === 'Finished');
      if (finishedCondition?.status === 'True') {
        status = 'Finished';
      }
      // Check for failure conditions
      else if (conditions.find(c => c.type === 'Failed')?.status === 'True') {
        status = 'Failed';
      }
      // Check for admission
      else if (conditions.find(c => c.type === 'Admitted')?.status === 'True') {
        status = 'Running';
      }
    }

    // Check if suspended
    if (workload.spec.suspend) {
      status = 'Suspended';
    }

    // Validate and return the status using the schema
    return jobStatusSchema.parse(status);
  }

  private getAdmittedTime(workload: KueueJob): string | undefined {
    const admittedCondition = workload.status?.conditions?.find(c => c.type === 'Admitted');
    return admittedCondition?.status === 'True' ? admittedCondition.lastTransitionTime : undefined;
  }
}

export const kubernetesClient = new KubernetesClient();