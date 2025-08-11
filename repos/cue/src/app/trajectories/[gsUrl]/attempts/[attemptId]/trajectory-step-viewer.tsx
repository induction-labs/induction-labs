/* eslint-disable @typescript-eslint/no-unsafe-call */
"use client";

import { useState, useEffect, useCallback } from "react";
import { useKeyPress } from 'ahooks';
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Switch } from "~/components/ui/switch";
import { Label } from "~/components/ui/label";
import { ChevronLeft, ChevronRight, Play, } from "lucide-react";
import { type TrajectorySteps } from "~/lib/schemas/trajectory";


interface TrajectoryStepViewerProps {
  steps: TrajectorySteps;
}

export function TrajectoryStepViewer({ steps }: TrajectoryStepViewerProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [showClickOverlay, setShowClickOverlay] = useState(true);
  const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);

  // Get image dimensions from base64 image data
  const getImageDimensions = useCallback((base64Data: string) => {
    return new Promise<{ width: number, height: number }>((resolve, reject) => {
      const img = document.createElement('img');
      img.onload = () => {
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = reject;
      img.src = `data:image/png;base64,${base64Data}`;
    });
  }, []);

  // Extract click coordinates from action field
  const extractClickCoordinates = (action: string) => {
    const pointRegex = /<point>(\d+)\s+(\d+)<\/point>/;
    const match = pointRegex.exec(action);
    if (match) {
      return {
        x: parseInt(match[1]!),
        y: parseInt(match[2]!)
      };
    }
    return null;
  };

  // Normalize coordinates to percentage of image dimensions
  const normalizeCoordinates = (coords: { x: number, y: number }) => {
    if (!imageDimensions) return null;
    return {
      x: (coords.x / imageDimensions.width) * 100,
      y: (coords.y / imageDimensions.height) * 100
    };
  };

  const currentStep = steps[currentStepIndex];
  const canGoPrevious = currentStepIndex > 0;
  const canGoNext = currentStepIndex < steps.length - 1;

  // Load image dimensions when step changes
  useEffect(() => {
    if (currentStep?.image) {
      getImageDimensions(currentStep.image)
        .then(setImageDimensions)
        .catch((error) => {
          console.warn('Failed to get image dimensions:', error);
          setImageDimensions(null);
        });
    } else {
      setImageDimensions(null);
    }
  }, [currentStep?.image, getImageDimensions]);

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

          <div className="flex items-center space-x-4">
            <div className="text-sm text-muted-foreground">
              {currentStepIndex + 1} / {steps.length}
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="click-overlay"
                checked={showClickOverlay}
                onCheckedChange={setShowClickOverlay}
              />
              <Label htmlFor="click-overlay" className="text-sm">
                Show clicks
              </Label>
            </div>
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
            <div className="border rounded-lg overflow-hidden bg-muted aspect-video justify-center flex items-center">
              {currentStep.image ? (
                <>
                  <div className="relative h-full">
                    {
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={`data:image/png;base64,${currentStep.image}`}
                        alt={`Step ${currentStep.step} screenshot`}
                        style={{
                          height: '100%',
                        }}
                        className="object-contain"

                      />
                    }
                    {/* Click overlay */}
                    {showClickOverlay && currentStep.action && (() => {
                      const coords = extractClickCoordinates(currentStep.action);
                      if (coords) {
                        const normalizedCoords = normalizeCoordinates(coords);
                        if (normalizedCoords) {
                          return (
                            <div
                              className="absolute rounded-full border-2 border-red-500 bg-red-500/20 pointer-events-none"
                              style={{
                                left: `${normalizedCoords.x}%`,
                                top: `${normalizedCoords.y}%`,
                                width: '10px',
                                height: '10px',
                                transform: 'translate(-50%, -50%)',
                              }}
                            />
                          );
                        }
                      }
                      return null;
                    })()}
                  </div>
                </>
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