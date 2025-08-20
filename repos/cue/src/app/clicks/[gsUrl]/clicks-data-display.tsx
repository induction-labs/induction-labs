"use client";

import Link from "next/link";
import { Badge } from "~/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { ClickableId } from "../../trajectories/clickable-id";
import { DataTable } from "~/components/data-table";
import { type ClickEvalRecord } from "~/lib/schemas/clicks";
import { useMemo } from "react";

interface ClicksDataDisplayProps {
  data: {
    records: ClickEvalRecord[];
    totalCount: number;
    filePath: string;
  };
  gsUrl: string;
}

export function ClicksDataDisplay({ data, gsUrl }: ClicksDataDisplayProps) {
  const getAccuracyColor = (isInBbox: boolean) => {
    return isInBbox ? "default" : "destructive";
  };

  const formatCoords = (x1: number, y1: number, x2: number, y2: number) => {
    return `(${x1},${y1})-(${x2},${y2})`;
  };

  const column_data = useMemo(() => data.records.map(x => ({
    id: x.input.id,
    instruction: x.input.instruction,
    gt_x1: x.input.x1,
    gt_y1: x.input.y1,
    gt_x2: x.input.x2,
    gt_y2: x.input.y2,
    pred_x: x.prediction_point ? x.prediction_point[0] : null,
    pred_y: x.prediction_point ? x.prediction_point[1] : null,
    is_in_bbox: x.is_in_bbox,
    pixel_distance: x.pixel_distance ?? null,
  })), [data.records]);
  type ClickColumn = typeof column_data[number];

  const columns = [
    {
      key: 'id' as keyof ClickColumn,
      label: 'ID',
      render: (record: ClickColumn) => {
        const value = record.id;
        const encodedResultId = encodeURIComponent(String(value));
        const resultHref = `/clicks/${gsUrl}/results/${encodedResultId}`;

        return (
          <Link href={resultHref} className="hover:underline">
            <ClickableId id={String(value)} />
          </Link>
        );
      },
    },
    {
      key: 'instruction' as keyof ClickColumn,
      label: 'Instruction',
      sortable: false,
      className: 'max-w-xl',
      render: (record: ClickColumn) => {
        const instruction = record.instruction;
        const id = record.id;
        const encodedResultId = encodeURIComponent(String(id));
        const resultHref = `/clicks/${gsUrl}/results/${encodedResultId}`;

        return (
          <Tooltip>
            <TooltipTrigger asChild>
              <Link href={resultHref}>
                <div className="line-clamp-2 cursor-pointer hover:text-primary transition-colors">
                  {String(instruction)}
                </div>
              </Link>
            </TooltipTrigger>
            <TooltipContent className="max-w-md">
              <p className="whitespace-pre-wrap">{String(instruction)}</p>
            </TooltipContent>
          </Tooltip>
        );
      },
    },
    {
      key: 'gt_x1' as keyof ClickColumn,
      label: 'Ground Truth Box',
      sortable: false,
      className: 'text-center font-mono text-sm',
      render: (record: ClickColumn) => (
        <span>{formatCoords(record.gt_x1, record.gt_y1, record.gt_x2, record.gt_y2)}</span>
      ),
    },
    {
      key: 'pred_x' as keyof ClickColumn,
      label: 'Predicted Coords',
      sortable: false,
      className: 'text-center font-mono text-sm',
      render: (record: ClickColumn) => {
        if (record.pred_x === null || record.pred_y === null) {
          return <span className="text-muted-foreground">No prediction</span>;
        }
        return <span>({record.pred_x},{record.pred_y})</span>;
      },
    },
    {
      key: 'is_in_bbox' as keyof ClickColumn,
      label: 'In Box',
      className: 'text-center',
      render: (record: ClickColumn) => {
        const value = record.is_in_bbox;
        return (
          <Badge variant={getAccuracyColor(Boolean(value))}>
            {Boolean(value) ? 'Yes' : 'No'}
          </Badge>
        );
      },
    },
    {
      key: 'pixel_distance' as keyof ClickColumn,
      label: 'Pixel Distance',
      className: 'text-center',
      render: (record: ClickColumn) => {
        const value = record.pixel_distance;
        if (value === null) {
          return <span className="text-muted-foreground">N/A</span>;
        }
        return <Badge variant="outline">{Number(value).toFixed(1)}px</Badge>;
      },
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-2xl font-bold">{data.totalCount}</div>
          <div className="text-sm text-muted-foreground">Total Clicks</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-2xl font-bold">
            {data.records.filter(r => r.is_in_bbox).length}
          </div>
          <div className="text-sm text-muted-foreground">In Bounding Box</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-2xl font-bold">
            {data.records.filter(r => !r.is_in_bbox).length}
          </div>
          <div className="text-sm text-muted-foreground">Outside Box</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-2xl font-bold">
            {(() => {
              const validDistances = data.records.filter(r => r.pixel_distance !== null);
              if (validDistances.length === 0) return 'N/A';
              const avgDistance = validDistances.reduce((sum, r) => sum + r.pixel_distance!, 0) / validDistances.length;
              return `${avgDistance.toFixed(1)}px`;
            })()}
          </div>
          <div className="text-sm text-muted-foreground">Avg Distance</div>
        </div>
      </div>

      <DataTable
        data={column_data}
        columns={columns}
        title="Click Evaluation Results"
        emptyMessage="No click evaluation data available"
        getRowKey={(record, index) => `${record.id}-${index}`}
        defaultSortField="id"
        defaultSortDirection="asc"
      />
    </div>
  );
}