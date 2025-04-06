import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertCircle, XCircle } from "lucide-react";
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
          <div className="space-y-2">
            {Array.isArray(results.issues) && results.issues.length > 0 ? (
              <div>
                <h4 className="font-medium mb-1">Detected Issues</h4>
                <ul className="list-disc list-inside space-y-1">
                  {results.issues.map((issue: any, index: number) => (
                    <li key={index} className="text-sm">
                      {issue.message || issue}
                      {issue.severity && (
                        <Badge className="ml-2" variant={
                          issue.severity === "high" ? "destructive" : 
                          issue.severity === "medium" ? "default" : "outline"
                        }>
                          {issue.severity}
                        </Badge>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="flex items-center text-green-600">
                <CheckCircle className="h-5 w-5 mr-2" />
                <span>No issues detected</span>
              </div>
            )}

            {results.userCount && (
              <div className="text-sm text-muted-foreground">
                Checked {results.userCount} user(s)
              </div>
            )}

            {results.recommendations && results.recommendations.length > 0 && (
              <div className="mt-3">
                <h4 className="font-medium mb-1">Recommendations</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  {results.recommendations.map((rec: string, index: number) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}

            {results.metrics && (
              <div className="mt-3">
                <h4 className="font-medium mb-1">Metrics</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {Object.entries(results.metrics).map(([key, value]: [string, any]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                      <span>{value}</span>
                    </div>
                  ))}
                </div>
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