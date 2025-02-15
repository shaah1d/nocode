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

// export default function Train() {
//   const [pythonFileContent, setPythonFileContent] = useState<string | null>(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);
//   const [modelName, setModelName] = useState<string>('');
//   const [isDialogOpen, setIsDialogOpen] = useState(false);
//   const [isCopied, setIsCopied] = useState(false);

//   const modelOptions = [
//     'XgBoost',
//     'KNN',
//     'RandomForestRegressor',
//     'DecisionTreesRegressor',
//     'LinearRegression',
//   ];

//   useEffect(() => {
//     const fetchData = async () => {
//       if (!modelName) {
//         return;
//       }

//       setLoading(true);
//       try {
//         const response = await fetch(`/api/python/${modelName}`);
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
//   }, [modelName]);

//   const handleModelChange = (event: any) => {
//     setError(null);
//     setModelName(event.target.value);
//   };

//   const handleCopyCode = async () => {
//     if (pythonFileContent) {
//       try {
//         await navigator.clipboard.writeText(pythonFileContent);
//         setIsCopied(true);
//         setTimeout(() => setIsCopied(false), 2000); // Reset after 2 seconds
//       } catch (err) {
//         console.error('Failed to copy code:', err);
//       }
//     }
//   };

//   return (
//     <Box sx={{ maxWidth: 600, margin: 'auto', padding: 4 }}>
//       <Typography variant="h4" gutterBottom>
//         Select a Model to View Its Python Code
//       </Typography>

//       <FormControl fullWidth sx={{ marginBottom: 2 }}>
//         <InputLabel id="model-select-label">Select Model</InputLabel>
//         <Select
//           labelId="model-select-label"
//           value={modelName}
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
//         <DialogContent className="max-w-3xl h-[80vh] max-h-[800px]">
//           <DialogHeader className="flex flex-col space-y-4">
//             <div className="flex justify-between items-center">
//               <DialogTitle className="text-xl font-bold">
//                 {modelName} Implementation
//               </DialogTitle>
//               <Button
//                 variant="outline"
//                 size="sm"
//                 onClick={handleCopyCode}
//                 className="h-8 px-2"
//               >
//                 {isCopied ? (
//                   <>
//                     <Check className="h-4 w-4 mr-2" />
                
//                   </>
//                 ) : (
//                   <>
//                     <Copy className="h-4 w-4 mr-2" />
                
//                   </>
//                 )}
//               </Button>
//             </div>
//             <ScrollArea className="h-full max-h-[calc(80vh-100px)]">
//               <DialogDescription asChild>
//                 <div className="relative">
//                   <SyntaxHighlighter
//                     language="python"
//                     style={tomorrow}
//                     customStyle={{
//                       margin: 0,
//                       borderRadius: '6px',
//                     }}
//                   >
//                     {pythonFileContent || ''}
//                   </SyntaxHighlighter>
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

interface ApiResponse {
  message: string;
  evaluation_metrics: {
    mean_squared_error: number;
    r2_score: number;
  };
  model_download_link: string;
}

export default function ModelTrainer() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [modelType, setModelType] = useState<string>("linear_regression");
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

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);
    formData.append("model_type", modelType);

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
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto min-h-96 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg p-6">
      <h1 className="text-2xl font-bold mb-4">Train Machine Learning Model</h1>

      {/* File Upload */}
      <FileUpload onChange={(files: File[]) => setFile(files[0])} />
      <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
        Upload your processed dataset (CSV file).
      </p>

      {/* Target Column Input */}
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

      {/* Model Type Selection */}
      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Select Model Type
        </label>
        <select
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
          className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
        >
          <option value="linear_regression">Linear Regression</option>
          <option value="decision_tree">Decision Tree</option>
          <option value="random_forest">Random Forest</option>
          <option value="svm">Support Vector Machine (SVM)</option>
        </select>
      </div>

      {/* Train Model Button */}
      <Button onClick={handleTrainModel} className="mt-4">
        Train Model
      </Button>

      {/* Response Dialog */}
      {response && (
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="mt-4">
              View Results and Download Model
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Model Training Results</DialogTitle>
              <DialogDescription>
                Review the evaluation metrics and download the trained model.
              </DialogDescription>
            </DialogHeader>

            {/* Evaluation Metrics */}
            <Card className="mt-4">
              <CardContent className="pt-6">
                <h3 className="text-lg font-semibold mb-4">Evaluation Metrics</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Mean Squared Error</span>
                    <span className="font-medium">
                      {response.evaluation_metrics.mean_squared_error.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">R² Score</span>
                    <span className="font-medium">
                      {response.evaluation_metrics.r2_score.toFixed(4)}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Download Model Button */}
            <div className="flex justify-end mt-4">
              <a
                href={`http://127.0.0.1:8000${response.model_download_link}`}
                download
                className="no-underline"
              >
                <Button className="flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Download Trained Model
                </Button>
              </a>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}