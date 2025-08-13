/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-floating-promises */
"use client";

import React, { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Badge } from "~/components/ui/badge";
import { ChevronLeft } from "lucide-react";
import { useClicksData } from "../../clicks-context";

interface ResultDetailPageProps {
  params: Promise<{ gsUrl: string; resultId: string }>;
}

export default function ResultDetailPage({ params }: ResultDetailPageProps) {
  const { data, error } = useClicksData();
  const [gsUrl, setGsUrl] = useState<string>("");
  const [resultId, setResultId] = useState<string>("");
  const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);

  useEffect(() => {
    void params.then(({ gsUrl, resultId }) => {
      setGsUrl(gsUrl);
      setResultId(decodeURIComponent(resultId));
    });
  }, [params]);
  const getImageDimensions = useCallback((imageUrl: string) => {
    return new Promise<{ width: number, height: number }>((resolve, reject) => {
      const img = document.createElement('img');
      img.onload = () => {
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = reject;
      img.src = imageUrl;
    });
  }, []);
  const resultData = data?.records.find(record => record.id === resultId);

  useEffect(() => {
    if (resultData?.image_path) {
      const imageUrl = convertImagePath(resultData.image_path);
      getImageDimensions(imageUrl)
        .then(setImageDimensions)
        .catch((error) => {
          console.warn('Failed to get image dimensions:', error);
          setImageDimensions(null);
        });
    } else {
      setImageDimensions(null);
    }
  }, [resultData?.image_path, getImageDimensions]);


  if (!gsUrl || !resultId) return null;

  const resultError = !resultData ? `Result with ID "${resultId}" not found` : null;

  // Get image dimensions from URL


  // Normalize coordinates to percentage of image dimensions
  const normalizeCoordinates = (coords: { x: number, y: number }) => {
    if (!imageDimensions) return null;
    return {
      x: (coords.x / imageDimensions.width) * 100,
      y: (coords.y / imageDimensions.height) * 100
    };
  };

  // Normalize bounding box coordinates
  const normalizeBoundingBox = (x1: number, y1: number, x2: number, y2: number) => {
    if (!imageDimensions) return null;
    return {
      left: (x1 / imageDimensions.width) * 100,
      top: (y1 / imageDimensions.height) * 100,
      width: ((x2 - x1) / imageDimensions.width) * 100,
      height: ((y2 - y1) / imageDimensions.height) * 100
    };
  };

  const getAccuracyColor = (isInBbox: boolean) => {
    return isInBbox ? "default" : "destructive";
  };

  const formatCoords = (x1: number, y1: number, x2: number, y2: number) => {
    return `x:[${x1}, ${x2}] y:[${y1}, ${y2}]`;
  };

  const convertImagePath = (imagePath: string) => {
    // Extract the last two parts of the path
    // e.g., `/tmp/clicks_dataset_eiv1svpl/showdown-clicks-dev/frames/396394CF-CCA3-49BB-8432-13E671950DEC/16.webp`
    // becomes `396394CF-CCA3-49BB-8432-13E671950DEC/16.webp`
    const pathParts = imagePath.split('/');
    if (pathParts.length >= 2) {
      const lastTwoParts = pathParts.slice(-2).join('/');
      return `https://storage.googleapis.com/click-eval/generalagents-showdown-clicks/showdown-clicks-dev/frames/${lastTwoParts}`;
    }
    return imagePath; // fallback to original if parsing fails
  };

  const parseRawResponse = (rawResponse: string) => {
    try {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const parsed = JSON.parse(rawResponse);
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment,
      const responseContent = parsed?.choices?.[0]?.message?.content;
      if (responseContent) {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        return { success: true, content: responseContent, fullResponse: parsed };
      }
      return { success: false, content: rawResponse, fullResponse: null };
    } catch {
      return { success: false, content: rawResponse, fullResponse: null };
    }
  };

  // Load image dimensions when result data changes

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <Button variant="outline" size="sm" asChild>
              <Link href={`/clicks/${gsUrl}`}>
                <ChevronLeft className="mr-2 h-4 w-4" />
                Back to Results
              </Link>
            </Button>
          </div>
          <div>
            <h1 className="text-4xl font-bold tracking-tight">Click Evaluation Result</h1>
            <p className="text-muted-foreground mt-2">
              ID: {resultId}
            </p>
          </div>
        </div>

        {/* Content */}
        {error || resultError ? (
          <div className="text-center py-12">
            <p className="text-destructive mb-4">{error ?? resultError}</p>
            <Button variant="outline" asChild>
              <Link href={`/clicks/${gsUrl}`}>Back to Results</Link>
            </Button>
          </div>
        ) : resultData ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Click Evaluation</span>
                <Badge variant={getAccuracyColor(resultData.is_in_bbox)}>
                  {resultData.is_in_bbox ? 'Success' : 'Failed'}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Task Info */}
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-2">Task</h3>
                <p className="text-sm">{resultData.instruction}</p>
              </div>

              {/* Main Content - Image and Stats */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Screenshot with overlays - 2/3 Width */}
                <div className="lg:col-span-2 space-y-2">
                  <h3 className="text-sm font-medium text-muted-foreground">Screenshot</h3>
                  <div className="relative border rounded-lg overflow-hidden bg-muted aspect-video flex items-center justify-center">
                    {resultData.image_path ? (
                      <>
                        <div className="relative h-full">
                          {
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                              src={convertImagePath(resultData.image_path)}
                              alt="Task screenshot"
                              style={{
                                height: '100%',
                              }}
                              className="object-contain"
                            />
                          }

                          {/* Ground Truth Bounding Box Overlay */}
                          {(() => {
                            const normalizedBox = normalizeBoundingBox(
                              resultData.gt_x1,
                              resultData.gt_y1,
                              resultData.gt_x2,
                              resultData.gt_y2
                            );
                            if (normalizedBox) {
                              return (
                                <div
                                  className="absolute border-2 border-green-500 bg-green-500/10 pointer-events-none"
                                  style={{
                                    left: `${normalizedBox.left}%`,
                                    top: `${normalizedBox.top}%`,
                                    width: `${normalizedBox.width}%`,
                                    height: `${normalizedBox.height}%`,
                                  }}
                                />
                              );
                            }
                            return null;
                          })()}

                          {/* Center point of bounding box */}
                          {resultData.center_coords && (() => {
                            const normalizedCenter = normalizeCoordinates({
                              x: resultData.center_coords[0],
                              y: resultData.center_coords[1]
                            });
                            if (normalizedCenter) {
                              return (
                                <div
                                  className="absolute rounded-full border-2 border-green-500 bg-green-500 pointer-events-none"
                                  style={{
                                    left: `${normalizedCenter.x}%`,
                                    top: `${normalizedCenter.y}%`,
                                    width: '8px',
                                    height: '8px',
                                    transform: 'translate(-50%, -50%)',
                                  }}
                                />
                              );
                            }
                            return null;
                          })()}

                          {/* Predicted Click Overlay */}
                          {resultData.pred_x !== null && resultData.pred_y !== null && (() => {
                            const normalizedPred = normalizeCoordinates({
                              x: resultData.pred_x,
                              y: resultData.pred_y
                            });
                            if (normalizedPred) {
                              return (
                                <div
                                  className="absolute rounded-full border-2 border-red-500 bg-red-500/20 pointer-events-none"
                                  style={{
                                    left: `${normalizedPred.x}%`,
                                    top: `${normalizedPred.y}%`,
                                    width: '16px',
                                    height: '16px',
                                    transform: 'translate(-50%, -50%)',
                                  }}
                                />
                              );
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

                  {/* Legend */}
                  <div className="flex items-center gap-6 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-green-500 bg-green-500/10"></div>
                      <span>Target Area</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span>Target Center</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full border-2 border-red-500 bg-red-500/20"></div>
                      <span>Predicted Click</span>
                    </div>
                  </div>
                </div>

                {/* Stats - 1/3 Width */}
                <div className="lg:col-span-1 space-y-4">
                  {/* Accuracy */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Result</h3>
                    <Badge variant={getAccuracyColor(resultData.is_in_bbox)}>
                      {resultData.is_in_bbox ? 'Click in Target' : 'Click Outside Target'}
                    </Badge>
                  </div>

                  {/* Coordinates */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Target Coordinates</h3>
                    <div className="bg-muted p-3 rounded-lg">
                      <p className="text-xs font-mono">{formatCoords(resultData.gt_x1, resultData.gt_y1, resultData.gt_x2, resultData.gt_y2)}</p>
                    </div>
                  </div>

                  {/* Predicted Click */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Predicted Click</h3>
                    <div className="bg-muted p-3 rounded-lg">
                      {resultData.pred_x !== null && resultData.pred_y !== null ? (
                        <p className="text-xs font-mono">({resultData.pred_x}, {resultData.pred_y})</p>
                      ) : (
                        <p className="text-xs text-muted-foreground">No prediction made</p>
                      )}
                    </div>
                  </div>

                  {/* Image Dimensions */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Image Dimensions</h3>
                    <div className="bg-muted p-3 rounded-lg">
                      {imageDimensions ? (
                        <p className="text-xs font-mono">{imageDimensions.width} × {imageDimensions.height}px</p>
                      ) : (
                        <p className="text-xs text-muted-foreground">Loading...</p>
                      )}
                    </div>
                  </div>

                  {/* Error Metrics */}
                  {(resultData.pixel_distance !== null || resultData.x_error !== null || resultData.y_error !== null) && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-muted-foreground">Error Metrics</h3>
                      <div className="bg-muted p-3 rounded-lg space-y-2">
                        {resultData.pixel_distance !== null && (
                          <div className="flex justify-between">
                            <span className="text-xs">Distance:</span>
                            <span className="text-xs font-mono">{resultData.pixel_distance.toFixed(1)}px</span>
                          </div>
                        )}
                        {resultData.x_error !== null && (
                          <div className="flex justify-between">
                            <span className="text-xs">X Error:</span>
                            <span className="text-xs font-mono">{resultData.x_error.toFixed(1)}px</span>
                          </div>
                        )}
                        {resultData.y_error !== null && (
                          <div className="flex justify-between">
                            <span className="text-xs">Y Error:</span>
                            <span className="text-xs font-mono">{resultData.y_error.toFixed(1)}px</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Performance */}
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Performance</h3>
                    <div className="bg-muted p-3 rounded-lg">
                      <div className="flex justify-between">
                        <span className="text-xs">Latency:</span>
                        <span className="text-xs font-mono">{resultData.latency_seconds.toFixed(2)}s</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Response Content */}
              {(() => {
                const parsedResponse = parseRawResponse(resultData.raw_response);
                if (parsedResponse.success) {
                  return (
                    <>
                      <div className="space-y-2">
                        <h3 className="text-sm font-medium text-muted-foreground">Response</h3>
                        <div className="bg-muted p-4 rounded-lg">
                          <pre className="text-xs whitespace-pre-wrap overflow-auto max-h-40">
                            {parsedResponse.content}
                          </pre>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <h3 className="text-sm font-medium text-muted-foreground">Raw Response</h3>
                        <div className="bg-muted p-4 rounded-lg">
                          <pre className="text-xs whitespace-pre-wrap overflow-auto max-h-40">
                            {JSON.stringify(parsedResponse.fullResponse, null, 2)}
                          </pre>
                        </div>
                      </div>
                    </>
                  );
                } else {
                  return (
                    <div className="space-y-2">
                      <h3 className="text-sm font-medium text-muted-foreground">Raw Response</h3>
                      <div className="bg-muted p-4 rounded-lg">
                        <pre className="text-xs whitespace-pre-wrap overflow-auto max-h-40">
                          {parsedResponse.content}
                        </pre>
                      </div>
                    </div>
                  );
                }
              })()}
            </CardContent>
          </Card>
        ) : (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Loading result data...</p>
          </div>
        )}
      </div>
    </main>
  );
}