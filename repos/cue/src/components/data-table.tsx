"use client";

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
import {
  TooltipProvider,
} from "~/components/ui/tooltip";
import { Button } from "~/components/ui/button";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";

type SortDirection = 'asc' | 'desc' | null;

interface SortableColumn<T> {
  key: keyof T;
  label: string;
  sortable?: boolean;
  className?: string;
  render?: (value: T[keyof T], record: T, index: number) => React.ReactNode;
}

interface DataTableProps<T> {
  data: T[];
  columns: SortableColumn<T>[];
  title: string;
  emptyMessage?: string;
  getRowKey?: (record: T, index: number) => string;
  defaultSortField?: keyof T;
  defaultSortDirection?: 'asc' | 'desc';
}

export function DataTable<T extends Record<string, unknown>>({
  data,
  columns,
  title,
  emptyMessage = "No data available",
  getRowKey = (_, index) => index.toString(),
  defaultSortField,
  defaultSortDirection = 'asc',
}: DataTableProps<T>) {
  const [sortField, setSortField] = useState<keyof T | null>(defaultSortField ?? null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(defaultSortField ? defaultSortDirection : null);

  const handleSort = (field: keyof T) => {
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

      // Handle string sorting
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        if (sortDirection === 'asc') {
          return aValue.localeCompare(bValue);
        } else {
          return bValue.localeCompare(aValue);
        }
      }

      // Handle numeric sorting
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        if (sortDirection === 'asc') {
          return aValue - bValue;
        } else {
          return bValue - aValue;
        }
      }

      // Handle mixed types - convert to string and sort
      const aString = String(aValue);
      const bString = String(bValue);
      if (sortDirection === 'asc') {
        return aString.localeCompare(bString);
      } else {
        return bString.localeCompare(aString);
      }
    });
  }, [data, sortField, sortDirection]);

  const getSortIcon = (field: keyof T) => {
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <TooltipProvider>
            <Table>
              <TableHeader>
                <TableRow>
                  {columns.map((column) => (
                    <TableHead key={String(column.key)} className={column.className}>
                      {column.sortable !== false ? (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-auto p-0 font-semibold"
                          onClick={() => handleSort(column.key)}
                        >
                          {column.label}
                          {getSortIcon(column.key)}
                        </Button>
                      ) : (
                        column.label
                      )}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedData.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={columns.length} className="text-center text-muted-foreground py-8">
                      {emptyMessage}
                    </TableCell>
                  </TableRow>
                ) : (
                  sortedData.map((record, index) => (
                    <TableRow key={getRowKey(record, index)}>
                      {columns.map((column) => (
                        <TableCell key={String(column.key)} className={column.className}>
                          {column.render ?
                            column.render(record[column.key], record, index) :
                            String(record[column.key] ?? '')
                          }
                        </TableCell>
                      ))}
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