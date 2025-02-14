'use client';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from 'react';
import { Select, MenuItem, FormControl, InputLabel, Box, Typography } from '@mui/material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';

export default function Hyper() {
  const [pythonFileContent, setPythonFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hyperModel, setHyperModel] = useState<string>('');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const modelOptions = [
    'XgBoostParameters',
    'KNNParameters',
    'RandomForestRegressorParameters',
    'DecisionTreesRegressorParameters',
    'LinearRegressionParameters',
  ];

  useEffect(() => {
    const fetchData = async () => {
      if (!hyperModel) {
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
        setIsDialogOpen(true);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [hyperModel]);

  const handleModelChange = (event: any) => {
    setError(null);
    setHyperModel(event.target.value);
  };

  const handleCopyCode = async () => {
    if (pythonFileContent) {
      try {
        await navigator.clipboard.writeText(pythonFileContent);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      } catch (err) {
        console.error('Failed to copy code:', err);
      }
    }
  };

  return (
    <Box sx={{ maxWidth: 600, margin: 'auto', padding: 4 }}>
      <Typography variant="h4" gutterBottom>
        Select a Model to View Its Python Code For Hyper Parameter Tuning
      </Typography>

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

      {loading && <Typography>Loading...</Typography>}
      {error && <Typography color="error">{error}</Typography>}

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-3xl w-[90vw] h-[80vh] max-h-[800px]">
          <DialogHeader className="flex flex-col space-y-4">
            <div className="flex justify-between items-center">
              <DialogTitle className="text-xl font-bold">
                {hyperModel} Implementation
              </DialogTitle>
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyCode}
                className="h-8 px-2"
              >
                {isCopied ? (
                  <>
                    <Check className="h-4 w-4 mr-2" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-2" />
                    Copy code
                  </>
                )}
              </Button>
            </div>
            <ScrollArea className="h-full max-h-[calc(80vh-100px)] w-full">
              <DialogDescription asChild>
                <div className="relative w-full overflow-hidden">
                  <div className="overflow-auto">
                    <SyntaxHighlighter
                      language="python"
                      style={tomorrow}
                      customStyle={{
                        margin: 0,
                        borderRadius: '6px',
                        maxWidth: 'none',
                        width: '100%'
                      }}
                      wrapLongLines={false}
                      showLineNumbers={true}
                    >
                      {pythonFileContent || ''}
                    </SyntaxHighlighter>
                  </div>
                </div>
              </DialogDescription>
            </ScrollArea>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </Box>
  );
}