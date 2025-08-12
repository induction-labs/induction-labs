"use client";

import { useState } from "react";
import { Input } from "~/components/ui/input";
import { Button } from "~/components/ui/button";
import { useRouter } from "next/navigation";
import { ArrowRight } from "lucide-react";
import { cn } from "~/lib/utils";

interface EditablePathInputProps {
  currentPath: string;
  basePath: string; // e.g., "/trajectories" or "/clicks"
  placeholder?: string;
  className?: string;
}

export function EditablePathInput({ 
  currentPath, 
  basePath, 
  placeholder = "Enter GCS path...",
  className = "font-mono text-sm"
}: EditablePathInputProps) {
  const router = useRouter();
  const [pathInput, setPathInput] = useState(currentPath);

  const isPathChanged = pathInput.trim() !== currentPath && pathInput.trim().length > 0;

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isPathChanged) {
      const encodedPath = encodeURIComponent(pathInput.trim());
      router.push(`${basePath}/${encodedPath}`);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handlePathSubmit(e);
    }
  };

  return (
    <form onSubmit={handlePathSubmit} className="flex gap-2">
      <Input
        value={pathInput}
        onChange={(e) => setPathInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={cn(
          className,
          "flex-1",
          isPathChanged && "border-orange-500 focus-visible:ring-orange-500"
        )}
      />
      <Button 
        type="submit" 
        size="sm" 
        disabled={!isPathChanged}
        className="px-3"
      >
        <ArrowRight className="h-4 w-4" />
      </Button>
    </form>
  );
}