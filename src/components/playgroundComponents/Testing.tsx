

import { useState } from "react";
import { useRouter } from "next/navigation";
import { FileUpload } from "@/components/ui/file-upload";
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
import { Input } from "../ui/input";
import { Loader2 , ChevronRight} from "lucide-react"; // For loading spinner

interface TestModelResponse {
  accuracy?: number;
  f1_score?: number;
  mean_squared_error?: number;
  r2_score?: number;
  predictions: any[];
}

export default function ModelTester() {
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [isTesting, setIsTesting] = useState<boolean>(false); // Loading state for testing
  const [testResponse, setTestResponse] = useState<TestModelResponse | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);

  const router = useRouter();

  const handleTestModel = async () => {
    if (!modelFile) {
      alert("Please upload a trained model (.joblib file).");
      return;
    }
    if (!testFile) {
      alert("Please upload a test dataset (CSV file).");
      return;
    }
    if (!targetColumn) {
      alert("Please specify the target column.");
      return;
    }
    setIsTesting(true); // Start loading
    const formData = new FormData();
    formData.append("model_file", modelFile);
    formData.append("test_data_file", testFile);
    formData.append("target_column", targetColumn);
    try {
      const res = await fetch("http://127.0.0.1:8000/test-model/", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error("Failed to test the model.");
      }
      const data: TestModelResponse = await res.json();
      setTestResponse(data);
    } catch (error) {
      console.error(error);
      alert("An error occurred while testing the model.");
    } finally {
      setIsTesting(false); // Stop loading
    }
  };

  return (
    <div className="h-screen w-full flex items-center justify-center bg-white p-4">
    <div className="w-full max-w-2xl space-y-4">
      <p className="text-sm text-gray-600">Upload your trained model and test dataset to evaluate performance.</p>

      <div className="space-y-4">
        <div>
          <FileUpload onChange={(files: File[]) => setModelFile(files[0])} />
          <p className="text-xs text-gray-500 mt-1">Upload your trained model (.joblib file).</p>
        </div>

        <div>
          <FileUpload onChange={(files: File[]) => setTestFile(files[0])} />
          <p className="text-xs text-gray-500 mt-1">Upload your test dataset (CSV file).</p>
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

        <Button onClick={handleTestModel} disabled={isTesting} className="w-full">
          {isTesting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Testing Model...
            </>
          ) : (
            <>
              Test Model
              <ChevronRight className="ml-2 h-4 w-4 mb-3" />
            </>
          )}
        </Button>
      </div>

      {testResponse && (
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="w-full mt-2">
              View Test Results
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Model Testing Results</DialogTitle>
              <DialogDescription>
                Review the evaluation metrics and predictions for the test dataset.
              </DialogDescription>
            </DialogHeader>

            <div className="mt-4 space-y-4">
              <div className="bg-gray-50 p-3 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">Evaluation Metrics</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Accuracy</span>
                    <span className="font-medium">{testResponse.accuracy.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">F1 Score</span>
                    <span className="font-medium">{testResponse.f1_score.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold mb-2">Predictions</h3>
                <div className="bg-gray-50 p-3 rounded-lg max-h-40 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr>
                        <th className="text-left font-medium text-gray-600 pb-2">Index</th>
                        <th className="text-left font-medium text-gray-600 pb-2">Prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {testResponse.predictions.map((pred, index) => (
                        <tr key={index}>
                          <td className="pr-4 py-1">{index + 1}</td>
                          <td>{pred}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  </div>
  );
}