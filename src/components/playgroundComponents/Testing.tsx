// 'use client';
// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogHeader,
//   DialogTitle,
// } from "@/components/ui/dialog";
// import { ScrollArea } from "@/components/ui/scroll-area";
// import { Button } from "@/components/ui/button";
// import { useEffect, useState } from 'react';
// import { Select, MenuItem, FormControl, InputLabel, Box, Typography } from '@mui/material';
// import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
// import { Copy, Check } from 'lucide-react';

// export default function Testing() {
//   const [pythonFileContent, setPythonFileContent] = useState<string | null>(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);
//   const [testModel, setTestModel] = useState<string>('');
//   const [isDialogOpen, setIsDialogOpen] = useState(false);
//   const [isCopied, setIsCopied] = useState(false);

//   const modelOptions = [
//     'XGBoostTesting',
//     'KNNTest',
//     'RandomForestTest',
//     'DecisionTreesTest',
//     'LinearRegressionTest',
//   ];

//   useEffect(() => {
//     const fetchData = async () => {
//       if (!testModel) {
//         return;
//       }

//       setLoading(true);
//       try {
//         const response = await fetch(`/api/testing/${testModel}`);
//         if (!response.ok) {
//           throw new Error('Failed to fetch Python file content');
//         }
//         const data = await response.json();
//         setPythonFileContent(data.content);
//         setIsDialogOpen(true);
//       } catch (err: any) {
//         setError(err.message);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchData();
//   }, [testModel]);

//   const handleModelChange = (event: any) => {
//     setError(null);
//     setTestModel(event.target.value);
//   };

//   const handleCopyCode = async () => {
//     if (pythonFileContent) {
//       try {
//         await navigator.clipboard.writeText(pythonFileContent);
//         setIsCopied(true);
//         setTimeout(() => setIsCopied(false), 2000);
//       } catch (err) {
//         console.error('Failed to copy code:', err);
//       }
//     }
//   };

//   return (
//     <Box sx={{ maxWidth: 600, margin: 'auto', padding: 4 }}>
//       <Typography variant="h4" gutterBottom>
//         Select a Model to View Its Python Code For Testing The Model
//       </Typography>

//       <FormControl fullWidth sx={{ marginBottom: 2 }}>
//         <InputLabel id="model-select-label">Select Model</InputLabel>
//         <Select
//           labelId="model-select-label"
//           value={testModel}
//           onChange={handleModelChange}
//           label="Select Model"
//         >
//           {modelOptions.map((option) => (
//             <MenuItem key={option} value={option}>
//               {option}
//             </MenuItem>
//           ))}
//         </Select>
//       </FormControl>

//       {loading && <Typography>Loading...</Typography>}
//       {error && <Typography color="error">{error}</Typography>}

//       <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
//         <DialogContent className="max-w-3xl w-[90vw] h-[80vh] max-h-[800px]">
//           <DialogHeader className="flex flex-col space-y-4">
//             <div className="flex justify-between items-center">
//               <DialogTitle className="text-xl font-bold">
//                 {testModel} Implementation
//               </DialogTitle>
//               <Button
//                 variant="outline"
//                 size="sm"
//                 onClick={handleCopyCode}
//                 className="h-8 px-2"
//               >
//                 {isCopied ? (
//                   <>
//                     <Check className="h-4 w-4 " />
                   
//                   </>
//                 ) : (
//                   <>
//                     <Copy className="h-4 w-4 " />
                 
//                   </>
//                 )}
//               </Button>
//             </div>
//             <ScrollArea className="h-full max-h-[calc(80vh-100px)] w-full">
//               <DialogDescription asChild>
//                 <div className="relative w-full overflow-hidden">
//                   <div className="overflow-auto">
//                     <SyntaxHighlighter
//                       language="python"
//                       style={tomorrow}
//                       customStyle={{
//                         margin: 0,
//                         borderRadius: '6px',
//                         maxWidth: 'none',
//                         width: '100%'
//                       }}
//                       wrapLongLines={false}
//                       showLineNumbers={true}
//                     >
//                       {pythonFileContent || ''}
//                     </SyntaxHighlighter>
//                   </div>
//                 </div>
//               </DialogDescription>
//             </ScrollArea>
//           </DialogHeader>
//         </DialogContent>
//       </Dialog>
//     </Box>
//   );
// }

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
import { Loader2 } from "lucide-react"; // For loading spinner

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
    <div className="w-full max-w-4xl mx-auto min-h-96 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg p-6">
      <h1 className="text-2xl font-bold mb-4">Test Trained Machine Learning Models</h1>

      {/* Test Model Section */}
      <h2 className="text-xl font-semibold mb-4">Test a Trained Model</h2>
      <FileUpload
        // label="Upload Trained Model (.joblib)"
        onChange={(files: File[]) => setModelFile(files[0])}
      />
      <FileUpload
        // label="Upload Test Dataset (CSV)"
        onChange={(files: File[]) => setTestFile(files[0])}
      />
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Target Column
        </label>
        <input
          type="text"
          placeholder="Enter target column name"
          value={targetColumn}
          onChange={(e) => setTargetColumn(e.target.value)}
          className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
        />
      </div>
      <Button onClick={handleTestModel} disabled={isTesting} className="mt-4">
        {isTesting ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : (
          "Test Model"
        )}
      </Button>

      {/* Response Dialog for Testing */}
      {testResponse && (
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="mt-4">
              View Test Results
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Model Testing Results</DialogTitle>
              <DialogDescription>
                Review the evaluation metrics and predictions for the uploaded
                test dataset.
              </DialogDescription>
            </DialogHeader>
            {/* Evaluation Metrics */}
            <Card className="mt-4">
              <CardContent className="pt-6">
                <h3 className="text-lg font-semibold mb-4">Evaluation Metrics</h3>
                <div className="space-y-2">
                  {testResponse.accuracy !== undefined ? (
                    <>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Accuracy</span>
                        <span className="font-medium">
                          {testResponse.accuracy?.toFixed(4) || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">F1 Score</span>
                        <span className="font-medium">
                          {testResponse.f1_score?.toFixed(4) || "N/A"}
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Mean Squared Error</span>
                        <span className="font-medium">
                          {testResponse.mean_squared_error?.toFixed(4) || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">R² Score</span>
                        <span className="font-medium">
                          {testResponse.r2_score?.toFixed(4) || "N/A"}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
            {/* Predictions */}
            <Card className="mt-4">
              <CardContent className="pt-6">
                <h3 className="text-lg font-semibold mb-4">Predictions</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        <th
                          scope="col"
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                        >
                          Prediction
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-black divide-y divide-gray-200 dark:divide-gray-700">
                      {testResponse.predictions.map((pred, index) => (
                        <tr key={index}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            {pred}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}