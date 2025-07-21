import * as k8s from '@kubernetes/client-node';
import { type KueueJob, type JobListItem, type JobStatus } from './types/kueue';

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
      const response = await this.customApi.listClusterCustomObject(
        'kueue.x-k8s.io',
        'v1beta1',
        'workloads'
      );

      const workloads = (response.body as any).items as KueueJob[];
      
      return workloads.map(workload => this.transformWorkloadToJobItem(workload));
    } catch (error) {
      console.error('Error fetching Kueue jobs:', error);
      throw new Error('Failed to fetch Kueue jobs from cluster');
    }
  }

  async getKueueJobsByNamespace(namespace: string): Promise<JobListItem[]> {
    try {
      const response = await this.customApi.listNamespacedCustomObject(
        'kueue.x-k8s.io',
        'v1beta1',
        namespace,
        'workloads'
      );

      const workloads = (response.body as any).items as KueueJob[];
      
      return workloads.map(workload => this.transformWorkloadToJobItem(workload));
    } catch (error) {
      console.error(`Error fetching Kueue jobs for namespace ${namespace}:`, error);
      throw new Error(`Failed to fetch Kueue jobs from namespace ${namespace}`);
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
    if (!workload.status?.conditions) {
      return 'Pending';
    }

    const conditions = workload.status.conditions;
    
    // Check for completion conditions
    const finishedCondition = conditions.find(c => c.type === 'Finished');
    if (finishedCondition?.status === 'True') {
      return 'Finished';
    }

    // Check for failure conditions
    const failedCondition = conditions.find(c => c.type === 'Failed');
    if (failedCondition?.status === 'True') {
      return 'Failed';
    }

    // Check for admission
    const admittedCondition = conditions.find(c => c.type === 'Admitted');
    if (admittedCondition?.status === 'True') {
      return 'Running';
    }

    // Check if suspended
    if (workload.spec.suspend) {
      return 'Suspended';
    }

    return 'Pending';
  }

  private getAdmittedTime(workload: KueueJob): string | undefined {
    const admittedCondition = workload.status?.conditions?.find(c => c.type === 'Admitted');
    return admittedCondition?.status === 'True' ? admittedCondition.lastTransitionTime : undefined;
  }
}

export const kubernetesClient = new KubernetesClient();