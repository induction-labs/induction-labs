"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface ClickableIdProps {
  id: string;
  truncateLength?: number;
}

export function ClickableId({ id, truncateLength = 5 }: ClickableIdProps) {
  const [copied, setCopied] = useState(false);

  const truncatedId = id.length > truncateLength ? `${id.substring(0, truncateLength)}...` : id;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="h-auto p-1 font-mono text-xs hover:bg-muted"
            onClick={handleCopy}
          >
            <span className="mr-1">{truncatedId}</span>
            {copied ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <Copy className="h-3 w-3 opacity-50" />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p className="font-mono text-xs">{id}</p>
          <p className="text-xs text-muted-foreground">Click to copy</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}