import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertCircle, XCircle, ThermometerIcon, Heart, Droplets, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface AgentResultDisplayProps {
  title: string;
  results: any;
  success: boolean;
  errorMessage?: string;
}

export const AgentResultDisplay: React.FC<AgentResultDisplayProps> = ({
  title,
  results,
  success,
  errorMessage
}) => {
  if (!results && !errorMessage) return null;

  // Function to get the appropriate status indicator
  const getStatusIndicator = (status: string) => {
    switch (status) {
      case "critical":
        return <span className="h-3 w-3 rounded-full bg-red-500"></span>;
      case "moderate":
        return <span className="h-3 w-3 rounded-full bg-amber-500"></span>;
      case "normal":
        return <span className="h-3 w-3 rounded-full bg-green-500"></span>;
      default:
        return <span className="h-3 w-3 rounded-full bg-gray-300"></span>;
    }
  };

  // Function to get badge variant based on severity
  const getBadgeVariant = (severity: string) => {
    switch (severity) {
      case "high":
        return "destructive";
      case "medium":
        return "default";
      default:
        return "outline";
    }
  };

  // Function to get badge for overall status
  const getOverallStatusBadge = (status: string) => {
    switch (status) {
      case "critical":
        return (
          <Badge variant="destructive" className="px-3 py-1">
            <AlertCircle className="h-4 w-4 mr-1" />
            Critical
          </Badge>
        );
      case "moderate":
        return (
          <Badge variant="default" className="bg-amber-500 text-white px-3 py-1">
            <AlertCircle className="h-4 w-4 mr-1" />
            Moderate Concern
          </Badge>
        );
      case "normal":
        return (
          <Badge variant="outline" className="bg-green-100 text-green-800 px-3 py-1">
            <CheckCircle className="h-4 w-4 mr-1" />
            Normal
          </Badge>
        );
      default:
        return null;
    }
  };

  // Function to get icon for metrics
  const getMetricIcon = (key: string) => {
    switch (key) {
      case "heart_rate":
        return <Heart className="h-4 w-4 text-red-500" />;
      case "temperature":
        return <ThermometerIcon className="h-4 w-4 text-amber-500" />;
      case "oxygen_saturation":
        return <Droplets className="h-4 w-4 text-blue-500" />;
      case "blood_pressure":
        return <Activity className="h-4 w-4 text-purple-500" />;
      default:
        return null;
    }
  };

  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">{title} Results</CardTitle>
          {success ? (
            <Badge variant="success" className="bg-green-600 text-white">
              <CheckCircle className="h-4 w-4 mr-1" />
              Success
            </Badge>
          ) : (
            <Badge variant="destructive">
              <XCircle className="h-4 w-4 mr-1" />
              Error
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {success && results ? (
          <div className="space-y-4">
            {/* Overall Status */}
            {results.overall_status && (
              <div className="mb-4 flex justify-between items-center">
                <span className="text-sm font-medium">Overall Status:</span>
                {getOverallStatusBadge(results.overall_status)}
              </div>
            )}

            {/* Issues / Anomalies */}
            {Array.isArray(results.issues) && results.issues.length > 0 ? (
              <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3">
                <h4 className="font-medium mb-2">Detected Issues</h4>
                <ul className="list-disc list-inside space-y-2">
                  {results.issues.map((issue: any, index: number) => (
                    <li key={index} className="text-sm">
                      {issue.message || issue}
                      {issue.severity && (
                        <Badge className="ml-2" variant={getBadgeVariant(issue.severity)}>
                          {issue.severity}
                        </Badge>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="flex items-center text-green-600 bg-green-50 dark:bg-green-900/20 p-3 rounded-md">
                <CheckCircle className="h-5 w-5 mr-2" />
                <span>No issues detected</span>
              </div>
            )}

            {/* Metrics with Status */}
            {results.metrics && Object.keys(results.metrics).length > 0 && (
              <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3">
                <h4 className="font-medium mb-2">Metrics</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {Object.entries(results.metrics).map(([key, value]: [string, any]) => {
                    const status = results.status_summary?.[key] || "normal";
                    return (
                      <div key={key} className="flex items-center justify-between p-2 border-b border-gray-100 dark:border-gray-800">
                        <div className="flex items-center">
                          {getMetricIcon(key)}
                          <span className="ml-2 text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                        </div>
                        <div className="flex items-center">
                          <span className="mr-2">{value}</span>
                          {getStatusIndicator(status)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {results.recommendations && results.recommendations.length > 0 && (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md p-3">
                <h4 className="font-medium mb-2">Recommendations</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  {results.recommendations.map((rec: string, index: number) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* User Count */}
            {results.userCount && (
              <div className="text-sm text-muted-foreground mt-2">
                Checked {results.userCount} user(s)
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center text-destructive">
            <AlertCircle className="h-5 w-5 mr-2" />
            <span>{errorMessage || "Unknown error occurred"}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AgentResultDisplay; 