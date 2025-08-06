"use client";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Button } from "~/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "~/components/ui/select";
import { TrajectoryTable } from "./trajectory-table";
import { Search, X } from "lucide-react";
import { type TrajectoryData } from "~/lib/schemas/trajectory";
import { useMemo } from "react";

enum SearchField {
  ALL = "all",
  TASK_ID = "task_id",
  ATTEMPT_ID = "attempt_id",
  INSTRUCTION = "instruction"
}

interface TrajectoryDataDisplayProps {
  trajectoryData: TrajectoryData;
  gsUrl?: string;
}

export function TrajectoryDataDisplay({ trajectoryData, gsUrl }: TrajectoryDataDisplayProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchField, setSearchField] = useState<SearchField>(SearchField.ALL);

  // Filter trajectory data based on search query and selected field
  const filteredData = useMemo(() => {
    if (!trajectoryData?.records || !searchQuery.trim()) {
      return trajectoryData?.records ?? [];
    }

    const query = searchQuery.toLowerCase().trim();
    return trajectoryData.records.filter((record) => {
      switch (searchField) {
        case SearchField.TASK_ID:
          return record.eval_task_id.toLowerCase().includes(query);
        case SearchField.ATTEMPT_ID:
          return record.attempt_id.toLowerCase().includes(query);
        case SearchField.INSTRUCTION:
          return record.instruction.toLowerCase().includes(query);
        case SearchField.ALL:
        default:
          return (
            record.eval_task_id.toLowerCase().includes(query) ||
            record.attempt_id.toLowerCase().includes(query) ||
            record.instruction.toLowerCase().includes(query)
          );
      }
    });
  }, [trajectoryData?.records, searchQuery, searchField]);

  const clearSearch = () => {
    setSearchQuery("");
  };

  // Calculate average reward for numeric values only
  const averageReward = useMemo(() => {
    const numericRewards = filteredData.filter(r => typeof r.reward === 'number');
    if (numericRewards.length === 0) return '0.00';
    const sum = numericRewards.reduce((sum: number, r: { reward: number | string }) => sum + (r.reward as number), 0);
    return (sum / numericRewards.length).toFixed(2);
  }, [filteredData]);

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Search Trajectories</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <Label htmlFor="search-input">Search Query</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="search-input"
                    placeholder={`Search in ${searchField === SearchField.ALL ? 'all fields' : searchField.replace('_', ' ')}...`}
                    value={searchQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-10"
                  />
                  {searchQuery && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8 p-0"
                      onClick={clearSearch}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="w-48">
                <Label htmlFor="search-field">Search In</Label>
                <Select value={searchField} onValueChange={(value: SearchField) => setSearchField(value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={SearchField.ALL}>All Fields</SelectItem>
                    <SelectItem value={SearchField.TASK_ID}>Task ID</SelectItem>
                    <SelectItem value={SearchField.ATTEMPT_ID}>Attempt ID</SelectItem>
                    <SelectItem value={SearchField.INSTRUCTION}>Instruction</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
          {searchQuery && (
            <p className="text-sm text-muted-foreground mt-2">
              Showing {filteredData.length} of {trajectoryData.totalCount} records
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>
            {searchQuery ? `Search Results (${filteredData.length})` : 'Trajectory Data Summary'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold">
                {searchQuery ? filteredData.length : trajectoryData.totalCount}
              </div>
              <div className="text-sm text-muted-foreground">
                {searchQuery ? 'Filtered Records' : 'Total Records'}
              </div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold">
                {filteredData.reduce((sum: number, r: { trajectory_length: number }) => sum + r.trajectory_length, 0)}
              </div>
              <div className="text-sm text-muted-foreground">Total Trajectory Length</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold">
                {averageReward}
              </div>
              <div className="text-sm text-muted-foreground">Average Reward (numeric only)</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Loaded from: <code className="bg-muted px-1 py-0.5 rounded text-xs">{trajectoryData.filePath}</code>
          </p>
        </CardContent>
      </Card>

      <TrajectoryTable data={filteredData} gsUrl={gsUrl} />
    </>
  );
}