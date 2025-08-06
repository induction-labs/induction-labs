import { Storage } from '@google-cloud/storage';
import { trajectoryRecordSchema, type TrajectoryRecord } from './schemas/trajectory';

class GCSClient {
  private storage: Storage;
  private projectId = 'induction-labs';

  constructor() {
    this.storage = new Storage({
      projectId: this.projectId,
    });
  }

  async readJSONLFile(gcsPath: string): Promise<TrajectoryRecord[]> {
    try {
      // Parse the GCS path to extract bucket and file name
      const match = gcsPath.match(/^gs:\/\/([^\/]+)\/(.+)$/);
      if (!match) {
        throw new Error('Invalid GCS path format. Expected: gs://bucket-name/path/to/file');
      }

      const [, bucketName, fileName] = match;
      
      // Get the file from GCS
      const file = this.storage.bucket(bucketName).file(fileName);
      
      // Check if file exists
      const [exists] = await file.exists();
      if (!exists) {
        throw new Error(`File not found: ${gcsPath}`);
      }

      // Download and read the file content
      const [content] = await file.download();
      const fileContent = content.toString('utf-8');

      // Parse JSONL (each line is a separate JSON object)
      const lines = fileContent.trim().split('\n');
      const records: TrajectoryRecord[] = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i]?.trim();
        if (!line) continue; // Skip empty lines

        try {
          const jsonObject = JSON.parse(line);
          // Validate each record using Zod schema
          const validatedRecord = trajectoryRecordSchema.parse(jsonObject);
          records.push(validatedRecord);
        } catch (error) {
          console.warn(`Failed to parse line ${i + 1} in ${gcsPath}:`, error);
          // Continue processing other lines instead of failing completely
        }
      }

      return records;
    } catch (error) {
      console.error(`Error reading JSONL file from GCS: ${gcsPath}`, error);
      if (error instanceof Error) {
        throw new Error(`Failed to read file from GCS: ${error.message}`);
      }
      throw new Error(`Failed to read file from GCS: ${gcsPath}`);
    }
  }

  async listFiles(bucketName: string, prefix?: string): Promise<string[]> {
    try {
      const [files] = await this.storage.bucket(bucketName).getFiles({
        prefix,
      });

      return files.map(file => `gs://${bucketName}/${file.name}`);
    } catch (error) {
      console.error(`Error listing files in bucket ${bucketName}:`, error);
      if (error instanceof Error) {
        throw new Error(`Failed to list files: ${error.message}`);
      }
      throw new Error(`Failed to list files in bucket ${bucketName}`);
    }
  }
}

export const gcsClient = new GCSClient();