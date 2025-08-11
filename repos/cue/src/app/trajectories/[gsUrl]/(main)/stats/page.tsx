"use client";

import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Badge } from "~/components/ui/badge";
import { BarChart3, TrendingUp, Target, Clock, Trophy } from "lucide-react";
import { useTrajectoryContext } from "../../trajectory-context";
import { useMemo } from "react";
import type { TrajectoryRecord } from "~/lib/schemas/trajectory";
import { DataFrame } from "data-forge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

/**
 * Draw k samples *with replacement* from arr.
 * @template T
 * @param {T[]} arr  – source array (length > 0)
 * @param {number} k – number of samples to draw
 * @returns {T[]}    – new array of length k
 */
function sampleWithReplacement<T>(arr: T[], k: number): T[] {
  if (!Array.isArray(arr) || arr.length === 0)
    throw new Error("Source array must be non-empty");

  const result = new Array<T>(k);
  for (let i = 0;i < k;i++) {
    // pick an index uniformly at random
    const idx = Math.floor(Math.random() * arr.length);
    result[i] = arr[idx]!;
  }
  return result;
}

// --- demo ---

// Example usage:
function calculateCIs(data: { eval_task_id: string, reward: number | string }[], top_k: number, nBootstrap = 10) {

  // const dfBinary = df.where(row =>
  //   // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access
  //   [0, 1, "0", "1"].includes(row.reward)
  // );


  const samples = [];
  for (let i = 0;i < nBootstrap;i++) {
    const shuffled = data.sort(() => Math.random() - 0.5);

    const rewards = new DataFrame(shuffled)
      .groupBy(r => r.eval_task_id)
      .select(group => {
        // 1. keep only the first k rows in this group

        const items = group.toArray();
        const sampled = sampleWithReplacement(items, top_k);
        const topreward = sampled.map(row => Number(row.reward)).reduce((a, b) => Math.max(a, b), 0);
        // <-- single row object

        return topreward;                                     // must return *one* object
      }).toArray();


    const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
    samples.push(mean);
  }

  samples.sort((a, b) => a - b);
  const meanAll = samples.reduce((a, b) => a + b, 0) / samples.length;
  const stdAll = Math.sqrt(
    samples.map(x => (x - meanAll) ** 2).reduce((a, b) => a + b, 0) / samples.length
  );
  const ciLower = samples[Math.floor(0.025 * samples.length)]!;
  const ciUpper = samples[Math.floor(0.975 * samples.length)]!;

  return { mean: meanAll, std: stdAll, ciLower, ciUpper };
}


export default function StatsPage() {
  const { trajectoryData } = useTrajectoryContext();

  // Group records by task ID for shared use across calculations
  const attemptsByTask = useMemo(() => {
    if (!trajectoryData?.records) {
      return new Map<string, TrajectoryRecord[]>();
    }

    const records = trajectoryData.records;
    const taskGroups = new Map<string, typeof records>();

    records.forEach(record => {
      if (!taskGroups.has(record.eval_task_id)) {
        taskGroups.set(record.eval_task_id, []);
      }
      taskGroups.get(record.eval_task_id)!.push(record);
    });

    return taskGroups;
  }, [trajectoryData]);

  const stats = useMemo(() => {
    if (!trajectoryData?.records) {
      return {
        totalTrajectories: 0,
        uniqueTasks: 0,
        numericRewards: [],
        stringRewards: [],
        avgTrajectoryLength: 0,
        avgReward: 0,
        successRate: 0,
        maxLength: 0,
        minLength: 0,
      };
    }

    const records = trajectoryData.records;
    const numericRewards = records.filter(r => typeof r.reward === 'number').map(r => r.reward as number);
    const stringRewards = records.filter(r => typeof r.reward === 'string');
    const trajectoryLengths = records.map(r => r.trajectory_length);

    return {
      totalTrajectories: records.length,
      uniqueTasks: attemptsByTask.size,
      numericRewards,
      stringRewards,
      avgTrajectoryLength: trajectoryLengths.reduce((sum, len) => sum + len, 0) / trajectoryLengths.length,
      avgReward: numericRewards.length > 0 ? numericRewards.reduce((sum, r) => sum + r, 0) / numericRewards.length : 0,
      successRate: numericRewards.length / records.length,
      maxLength: Math.max(...trajectoryLengths),
      minLength: Math.min(...trajectoryLengths),
    };
  }, [trajectoryData, attemptsByTask]);

  // Calculate best@k performance separately for better performance
  const bestAtK = useMemo(() => {
    if (attemptsByTask.size === 0) {
      return {
        best1: 0,
        best2: 0,
        best3: 0,
        best5: 0,
        best10: 0,
      };
    }

    // Calculate best@k performance
    const calculateBestAtK = (k: number) => {
      let totalReward = 0;

      attemptsByTask.forEach((taskRecords) => {
        // Sort by attempt ID
        const sortedRecords = taskRecords.sort((a, b) => {
          return a.attempt_id.localeCompare(b.attempt_id);
        });

        // Get the highest reward from the first k attempts
        const topK = sortedRecords.slice(0, k);
        const rewards = topK.map(record =>
          typeof record.reward === 'number' ? record.reward : 0
        );
        const maxReward = Math.max(...rewards);
        totalReward += maxReward;
      });

      return attemptsByTask.size > 0 ? totalReward / attemptsByTask.size : 0;
    };

    return {
      best1: calculateBestAtK(1),
      best2: calculateBestAtK(2),
      best3: calculateBestAtK(3),
      best5: calculateBestAtK(5),
      best10: calculateBestAtK(10),
    };
  }, [attemptsByTask]);

  // Calculate confidence intervals for metrics
  const confidenceIntervals = useMemo(() => {
    if (!trajectoryData?.records || attemptsByTask.size === 0) {
      return {
        avgReward: { mean: 0, std: 0, ciLower: 0, ciUpper: 0 },
        best1: { mean: 0, std: 0, ciLower: 0, ciUpper: 0 },
        best2: { mean: 0, std: 0, ciLower: 0, ciUpper: 0 },
        best3: { mean: 0, std: 0, ciLower: 0, ciUpper: 0 },
        best5: { mean: 0, std: 0, ciLower: 0, ciUpper: 0 },
        best10: { mean: 0, std: 0, ciLower: 0, ciUpper: 0 },
      };
    }

    // Helper function to calculate CI for best@k
    const calculateBestAtKCI = (k: number) => {
      const taskRewards: { reward: number, eval_task_id: string }[] = [];

      attemptsByTask.forEach((taskRecords) => {

        for (const record of taskRecords) {
          const reward = typeof record.reward === 'number' ? record.reward : 0;
          taskRewards.push({ reward, eval_task_id: record.eval_task_id });
        }
      });

      return calculateCIs(taskRewards, k);
    };

    // Calculate CI for average reward (using all numeric rewards)
    const numericRewards = trajectoryData.records
      .filter(r => typeof r.reward === 'number') as { eval_task_id: string, reward: number }[];

    let avgRewardCI = { mean: 0, std: 0, ciLower: 0, ciUpper: 0 };
    if (numericRewards.length > 0) {
      avgRewardCI = calculateCIs(numericRewards, 1);
    }

    return {
      avgReward: avgRewardCI,
      best1: calculateBestAtKCI(1),
      best2: calculateBestAtKCI(2),
      best3: calculateBestAtKCI(3),
      best5: calculateBestAtKCI(5),
      best10: calculateBestAtKCI(10),
    };
  }, [trajectoryData, attemptsByTask]);

  // Calculate trajectory length distribution for histogram with good/bad breakdown
  const trajectoryLengthDistribution = useMemo(() => {
    if (!trajectoryData?.records) {
      return [];
    }

    const records = trajectoryData.records;
    const lengths = records.map(r => r.trajectory_length);
    const minLength = Math.min(...lengths);
    const maxLength = Math.max(...lengths);

    // Create 10 bins
    const numBins = 10;
    const binSize = (maxLength - minLength) / numBins;

    const bins = Array.from({ length: numBins }, (_, i) => ({
      name: `${Math.round(minLength + i * binSize)}-${Math.round(minLength + (i + 1) * binSize)}`,
      good: 0, // reward > 0
      bad: 0,  // reward = 0 or string (failed)
      total: 0,
      binStart: minLength + i * binSize,
      binEnd: minLength + (i + 1) * binSize,
    }));

    // Assign records to bins and categorize by reward
    records.forEach(record => {
      const length = record.trajectory_length;
      const binIndex = Math.min(
        Math.floor((length - minLength) / binSize),
        numBins - 1
      );

      const isGood = typeof record.reward === 'number' && record.reward > 0;

      if (isGood) {
        bins[binIndex]!.good++;
      } else {
        bins[binIndex]!.bad++;
      }
      bins[binIndex]!.total++;
    });

    return bins;
  }, [trajectoryData]);

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Trajectories</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalTrajectories}</div>
            <p className="text-xs text-muted-foreground">
              {stats.uniqueTasks} unique tasks
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(stats.successRate * 100).toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              {stats.numericRewards.length} successful / {stats.totalTrajectories} total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Reward</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgReward.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">
              CI: [{confidenceIntervals.avgReward.ciLower.toFixed(3)}, {confidenceIntervals.avgReward.ciUpper.toFixed(3)}]
            </p>
            <p className="text-xs text-muted-foreground">
              From {stats.numericRewards.length} numeric rewards
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Steps</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgTrajectoryLength.toFixed(1)}</div>
            <p className="text-xs text-muted-foreground">
              Range: {stats.minLength} - {stats.maxLength}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Best@k Performance */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Best@k Performance</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best@1</CardTitle>
              <Trophy className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{confidenceIntervals.best1.mean.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">
                CI: [{confidenceIntervals.best1.ciLower.toFixed(3)}, {confidenceIntervals.best1.ciUpper.toFixed(3)}]
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best@2</CardTitle>
              <Trophy className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{confidenceIntervals.best2.mean.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">
                CI: [{confidenceIntervals.best2.ciLower.toFixed(3)}, {confidenceIntervals.best2.ciUpper.toFixed(3)}]
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best@3</CardTitle>
              <Trophy className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{confidenceIntervals.best3.mean.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">
                CI: [{confidenceIntervals.best3.ciLower.toFixed(3)}, {confidenceIntervals.best3.ciUpper.toFixed(3)}]
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best@5</CardTitle>
              <Trophy className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{confidenceIntervals.best5.mean.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">
                CI: [{confidenceIntervals.best5.ciLower.toFixed(3)}, {confidenceIntervals.best5.ciUpper.toFixed(3)}]
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best@10</CardTitle>
              <Trophy className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{confidenceIntervals.best10.mean.toFixed(3)}</div>
              <p className="text-xs text-muted-foreground">
                CI: [{confidenceIntervals.best10.ciLower.toFixed(3)}, {confidenceIntervals.best10.ciUpper.toFixed(3)}]
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Detailed Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Reward Distribution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {stats.numericRewards.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Numeric Rewards</h4>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <span className="text-muted-foreground">Min:</span>{" "}
                    <Badge variant="outline">{Math.min(...stats.numericRewards).toFixed(3)}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Max:</span>{" "}
                    <Badge variant="outline">{Math.max(...stats.numericRewards).toFixed(3)}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Count:</span>{" "}
                    <Badge variant="outline">{stats.numericRewards.length}</Badge>
                  </div>
                </div>
              </div>
            )}

            {stats.stringRewards.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Failure Types</h4>
                <div className="flex flex-wrap gap-2">
                  {Array.from(new Set(stats.stringRewards.map(r => r.reward as string))).map((failureType) => {
                    const count = stats.stringRewards.filter(r => r.reward === failureType).length;
                    return (
                      <Badge key={failureType} variant="destructive">
                        {failureType} ({count})
                      </Badge>
                    );
                  })}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Trajectory Length Distribution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-sm text-muted-foreground">Shortest</span>
                <div className="text-2xl font-bold">{stats.minLength}</div>
              </div>
              <div>
                <span className="text-sm text-muted-foreground">Longest</span>
                <div className="text-2xl font-bold">{stats.maxLength}</div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Average Length</span>
                <span className="font-mono">{stats.avgTrajectoryLength.toFixed(1)} steps</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full"
                  style={{
                    width: `${Math.min((stats.avgTrajectoryLength / stats.maxLength) * 100, 100)}%`
                  }}
                />
              </div>
            </div>

            {/* Histogram */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Distribution (10 bins)</h4>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={trajectoryLengthDistribution}
                    margin={{
                      top: 20,
                      right: 30,
                      left: 20,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis
                      dataKey="name"
                      className="text-xs fill-muted-foreground"
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis className="text-xs fill-muted-foreground" />
                    <Legend
                      formatter={(value: string) =>
                        value === 'good' ? 'Good (reward > 0)' : 'Bad (reward ≤ 0)'
                      }
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--popover))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '6px',
                      }}
                      labelStyle={{ color: 'hsl(var(--popover-foreground))' }}
                      formatter={(value: number, name: string) => [
                        value,
                        name === 'good' ? 'Good (reward > 0)' : 'Bad (reward ≤ 0)'
                      ]}
                    />
                    <Bar
                      dataKey="bad"
                      stackId="a"
                      fill="#ef4444"
                      name="bad"
                    />
                    <Bar
                      dataKey="good"
                      stackId="a"
                      fill="#22c55e"
                      radius={[2, 2, 0, 0]}
                      name="good"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}