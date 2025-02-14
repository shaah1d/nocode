// app/page.tsx
'use client'; // This is a client component

import { useEffect, useState } from 'react';
import { Select, MenuItem, FormControl, InputLabel, Box, Typography } from '@mui/material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism'; // A theme for syntax highlighting

export default function Home() {
  const [pythonFileContent, setPythonFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hyperModel, setHyperModel] = useState<string>(''); // No default model name

  // List of available models
  const modelOptions = [
    'XgBoostParameters',
    //'AdaBoost',
   // 'CatBoost',
    //'LassoRegression',
    //'RidgeRegression',
    'KNNParameters',
    //'SVR',
    'RandomForestRegressorParameters',
    'DecisionTreesRegressorParameters',
    //'ElasticNet',
    'LinearRegressionParameters',
  ];

  useEffect(() => {
    const fetchData = async () => {
      if (!hyperModel) {
        // Do not fetch data if no model is selected
        return;
      }

      setLoading(true);
      try {
        const response = await fetch(`/api/hyper/${hyperModel}`);
        if (!response.ok) {
          throw new Error('Failed to fetch Python file content');
        }
        const data = await response.json();
        setPythonFileContent(data.content);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [hyperModel]); // Re-run the effect when modelName changes

  const handleModelChange = (event: any) => {
    setError(null)
    setHyperModel(event.target.value); // Update the selected model name
  };

  return (
    <Box sx={{ maxWidth: 600, margin: 'auto', padding: 4 }}>
      <Typography variant="h4" gutterBottom>
        Select a Model to View Its Python Code For Hyper Parameter Tuning
      </Typography>

      {/* Dropdown Menu for Model Selection */}
      <FormControl fullWidth sx={{ marginBottom: 2 }}>
        <InputLabel id="model-select-label">Select Model</InputLabel>
        <Select
          labelId="model-select-label"
          value={hyperModel}
          onChange={handleModelChange}
          label="Select Model"
        >
          {modelOptions.map((option) => (
            <MenuItem key={option} value={option}>
              {option}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Loading State */}
      {loading && <Typography>Loading...</Typography>}

      {/* Error State */}
      {error && <Typography color="error">{error}</Typography>}

      {/* Display Python File Content */}
      {pythonFileContent && (
        <SyntaxHighlighter
        language="python" // Specify the language for syntax highlighting
        style={tomorrow} // Choose a theme (e.g., tomorrow, vs, darcula, etc.)
        customStyle={{
          backgroundColor: '#f5f5f5',
          padding: '1rem',
          borderRadius: '8px',
        }}
      >
            {pythonFileContent}
        </SyntaxHighlighter>
      )}
    </Box>
  );
}