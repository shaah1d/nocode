
import { useState } from "react";
import { useRouter } from "next/navigation";
import { FileUpload } from "@/components/ui/file-upload";
import { Input } from "../ui/input";
import { ChevronRight } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Download } from "lucide-react";
import { Loader2 } from "lucide-react"; // For loading spinner

interface ApiResponse {
  best_model_name: string;
  best_model_score: number;
  evaluation_metrics_all_models: Record<
    string,
    {
      accuracy?: number; // For classification
      f1_score?: number; // For classification
      mean_squared_error?: number; // For regression
      r2_score?: number; // For regression
    }
  >;
  best_model_download_link: string;
}

export default function ModelTrainer() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false); // Loading state
  const router = useRouter();

  const handleTrainModel = async () => {
    if (!file) {
      alert("Please upload a file first.");
      return;
    }
    if (!targetColumn) {
      alert("Please specify the target column.");
      return;
    }

    setIsLoading(true); // Start loading
    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);

    try {
      const res = await fetch("http://127.0.0.1:8000/train-model/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to train the model.");
      }

      const data: ApiResponse = await res.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
      alert("An error occurred while training the model.");
    } finally {
      setIsLoading(false); // Stop loading
    }
  };

  return (
    <div className="h-screen w-full flex items-center justify-center bg-white p-4">
    <div className="w-full max-w-2xl space-y-4">
      <p className="text-sm text-gray-600">Upload your dataset and configure your model training parameters.</p>

      <div className="space-y-4">
        <div>
          <FileUpload onChange={(files: File[]) => setFile(files[0])} />
          <p className="text-xs text-gray-500 mt-1">Upload your processed dataset (CSV file).</p>
        </div>

        <div>
          <label htmlFor="target-column" className="text-sm font-medium text-gray-700 block mb-1">
            Target Column
          </label>
          <Input
            id="target-column"
            placeholder="Enter target column name"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
          />
        </div>

        <Button onClick={handleTrainModel} disabled={isLoading} className="w-full">
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Training Models...
            </>
          ) : (
            <>
              Train Models
              <ChevronRight className="ml-2 h-4 w-4" />
            </>
          )}
        </Button>
      </div>

      {response && (
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="w-full mt-2">
              View Results and Download Best Model
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Model Training Results</DialogTitle>
              <DialogDescription>Review the evaluation metrics and download the best model.</DialogDescription>
            </DialogHeader>

            <div className="mt-4 space-y-4">
              <div className="bg-gray-50 p-3 rounded-lg">
                <h3 className="text-sm font-semibold">Best Model: {response.best_model_name}</h3>
                <p className="text-xs text-gray-600">Score: {response.best_model_score.toFixed(4)}</p>
              </div>

              <div>
                <h3 className="text-sm font-semibold mb-2">Evaluation Metrics</h3>
                <div className="space-y-2">
                  {Object.entries(response.evaluation_metrics_all_models).map(
                    ([modelName, metrics]: [string, any]) => (
                      <div key={modelName} className="bg-gray-50 p-2 rounded-lg">
                        <h4 className="text-xs font-medium mb-1">{modelName}</h4>
                        <div className="grid grid-cols-2 gap-1 text-xs">
                          {Object.entries(metrics).map(([metricName, value]: [string, number]) => (
                            <div key={metricName} className="flex justify-between">
                              <span className="text-gray-600">{metricName}</span>
                              <span className="font-medium">{value.toFixed(4)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ),
                  )}
                </div>
              </div>

              <div className="flex justify-end">
                <a
                  href={`http://127.0.0.1:8000${response.best_model_download_link}`}
                  download
                  className="no-underline"
                >
                  <Button size="sm" className="flex items-center gap-1">
                    <Download className="w-3 h-3" />
                    Download Best Model
                  </Button>
                </a>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  </div>
  );
}