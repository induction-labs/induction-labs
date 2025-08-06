import { useState, useMemo } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "~/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Button } from "~/components/ui/button";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { ClickableId } from "./clickable-id";
import { type TrajectoryRecord } from "~/lib/schemas/trajectory";

type SortField = 'eval_task_id' | 'attempt_id' | 'trajectory_length' | 'reward';
type SortDirection = 'asc' | 'desc' | null;

interface TrajectoryTableProps {
  data: TrajectoryRecord[];
}

export function TrajectoryTable({ data }: TrajectoryTableProps) {
  const [sortField, setSortField] = useState<SortField | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(null);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Cycle through: null -> asc -> desc -> null
      if (sortDirection === null) {
        setSortDirection('asc');
      } else if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else {
        setSortField(null);
        setSortDirection(null);
      }
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const sortedData = useMemo(() => {
    if (!sortField || !sortDirection) {
      return data;
    }

    return [...data].sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];

      // Handle string sorting for IDs
      if (sortField === 'eval_task_id' || sortField === 'attempt_id') {
        if (sortDirection === 'asc') {
          return String(aValue).localeCompare(String(bValue));
        } else {
          return String(bValue).localeCompare(String(aValue));
        }
      }

      // Handle numeric sorting for trajectory_length and reward
      if (sortField === 'reward') {
        // Handle mixed string/number rewards
        const aIsNumber = typeof aValue === 'number';
        const bIsNumber = typeof bValue === 'number';
        
        // Numbers come before strings in sorting
        if (aIsNumber && !bIsNumber) return sortDirection === 'asc' ? -1 : 1;
        if (!aIsNumber && bIsNumber) return sortDirection === 'asc' ? 1 : -1;
        
        // Both are strings - sort alphabetically
        if (!aIsNumber && !bIsNumber) {
          if (sortDirection === 'asc') {
            return String(aValue).localeCompare(String(bValue));
          } else {
            return String(bValue).localeCompare(String(aValue));
          }
        }
        
        // Both are numbers - sort numerically
        if (sortDirection === 'asc') {
          return Number(aValue) - Number(bValue);
        } else {
          return Number(bValue) - Number(aValue);
        }
      }
      
      // Handle numeric sorting for trajectory_length
      if (sortDirection === 'asc') {
        return Number(aValue) - Number(bValue);
      } else {
        return Number(bValue) - Number(aValue);
      }
    });
  }, [data, sortField, sortDirection]);

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="ml-2 h-4 w-4" />;
    }
    if (sortDirection === 'asc') {
      return <ArrowUp className="ml-2 h-4 w-4" />;
    }
    if (sortDirection === 'desc') {
      return <ArrowDown className="ml-2 h-4 w-4" />;
    }
    return <ArrowUpDown className="ml-2 h-4 w-4" />;
  };

  const getRewardColor = (reward: string | number) => {
    if (typeof reward === 'string') {
      return "destructive"; // String rewards (like "Internal Fail") are errors
    }
    if (reward >= 0.8) return "default";
    if (reward >= 0.5) return "secondary";
    return "destructive";
  };

  const formatReward = (reward: string | number) => {
    if (typeof reward === 'string') {
      return reward; // Return the string as-is
    }
    return reward.toFixed(3);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Trajectory Records</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <TooltipProvider>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-auto p-0 font-semibold"
                      onClick={() => handleSort('eval_task_id')}
                    >
                      Task ID
                      {getSortIcon('eval_task_id')}
                    </Button>
                  </TableHead>
                  <TableHead>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-auto p-0 font-semibold"
                      onClick={() => handleSort('attempt_id')}
                    >
                      Attempt ID
                      {getSortIcon('attempt_id')}
                    </Button>
                  </TableHead>
                  <TableHead>Instruction</TableHead>
                  <TableHead className="text-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-auto p-0 font-semibold"
                      onClick={() => handleSort('trajectory_length')}
                    >
                      Trajectory Length
                      {getSortIcon('trajectory_length')}
                    </Button>
                  </TableHead>
                  <TableHead className="text-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-auto p-0 font-semibold"
                      onClick={() => handleSort('reward')}
                    >
                      Reward
                      {getSortIcon('reward')}
                    </Button>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedData.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                      No trajectory data available
                    </TableCell>
                  </TableRow>
                ) : (
                  sortedData.map((record, index) => (
                    <TableRow key={`${record.eval_task_id}-${record.attempt_id}-${index}`}>
                      <TableCell>
                        <ClickableId id={record.eval_task_id} />
                      </TableCell>
                      <TableCell>
                        <ClickableId id={record.attempt_id} />
                      </TableCell>
                      <TableCell className="max-w-xl">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="line-clamp-2 cursor-help">
                              {record.instruction}
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-md">
                            <p className="whitespace-pre-wrap">{record.instruction}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge variant="outline">{record.trajectory_length}</Badge>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge variant={getRewardColor(record.reward)}>
                          {formatReward(record.reward)}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TooltipProvider>
        </div>
      </CardContent>
    </Card>
  );
}