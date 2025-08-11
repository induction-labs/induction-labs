/* eslint-disable @typescript-eslint/no-unsafe-call */
"use client";

import { useState } from "react";
import { useKeyPress } from 'ahooks';
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { ChevronLeft, ChevronRight, Play, Pause } from "lucide-react";
import { type TrajectorySteps } from "~/lib/schemas/trajectory";
import Image from "next/image";

interface TrajectoryStepViewerProps {
  steps: TrajectorySteps;
}

export function TrajectoryStepViewer({ steps }: TrajectoryStepViewerProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);


  const currentStep = steps[currentStepIndex];
  const canGoPrevious = currentStepIndex > 0;
  const canGoNext = currentStepIndex < steps.length - 1;

  const handlePrevious = () => {
    if (canGoPrevious) {
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };

  const handleNext = () => {
    if (canGoNext) {
      setCurrentStepIndex(currentStepIndex + 1);
    }
  };

  // Add keyboard navigation
  useKeyPress('leftarrow', () => {
    handlePrevious();
  });

  useKeyPress('rightarrow', () => {
    handleNext();
  });
  if (!steps || steps.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Play className="mr-2 h-5 w-5" />
            Trajectory Steps
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <p>No trajectory steps available</p>
          </div>
        </CardContent>
      </Card>
    );
  }


  if (!currentStep) {
    return null;
  }


  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Play className="mr-2 h-5 w-5" />
            Trajectory Steps
          </div>
          <Badge variant="outline">
            Step {currentStep.step} of {steps.length}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Step Navigation */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrevious}
            disabled={!canGoPrevious}
          >
            <ChevronLeft className="mr-2 h-4 w-4" />
            Previous
          </Button>

          <div className="text-sm text-muted-foreground">
            {currentStepIndex + 1} / {steps.length}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleNext}
            disabled={!canGoNext}
          >
            Next
            <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </div>

        {/* Step Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Screenshot - 2/3 Width */}
          <div className="lg:col-span-2 space-y-2">
            <h3 className="text-sm font-medium text-muted-foreground">Screenshot</h3>
            <div className="relative border rounded-lg overflow-hidden bg-muted aspect-video">
              {currentStep.image ? (
                <Image
                  src={`data:image/png;base64,${currentStep.image}`}
                  alt={`Step ${currentStep.step} screenshot`}
                  fill
                  className="object-contain"
                  unoptimized={true}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  No image available
                </div>
              )}
            </div>
          </div>

          {/* Action and Text - 1/3 Width */}
          <div className="lg:col-span-1 space-y-4">
            {/* Action */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-muted-foreground">Action</h3>
              <div className="bg-muted p-3 rounded-lg">
                <code className="text-xs whitespace-pre-wrap">{currentStep.action}</code>
              </div>
            </div>

            {/* Text */}
            {currentStep.text && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground">Description</h3>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="text-sm">{currentStep.text}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Step Progress */}
        <div className="w-full bg-muted rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all duration-300"
            style={{ width: `${((currentStepIndex + 1) / steps.length) * 100}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}