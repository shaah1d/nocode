"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import FileUploadIcon from '@mui/icons-material/FileUpload';
import { FileUpload } from "@/components/ui/file-upload";

interface ApiResponse {
  message: string;
  summary_statistics: Record<string, any>;
  download_link: string;
}

export default function Uploader() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const router = useRouter();

  const handleUpload = async (files: File[]) => {
    if (files.length === 0) {
      alert("Please select a file");
      return;
    }

    // Set the first file from the array
    const selectedFile = files[0];
    setFile(selectedFile);

    // Create FormData and append the file
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("http://127.0.0.1:8000/process-csv/", {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) {
        throw new Error("Failed to process CSV");
      }
      
      const data: ApiResponse = await res.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
      alert("An error occurred while processing the file");
    }
  };

  const handleNextStep = () => {
    router.push("/readpy");
  };

  return (
    <div className="w-full max-w-4xl mx-auto min-h-96 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg">
      <FileUpload onChange={handleUpload} />

      {response && (
        <div style={{ marginTop: "2rem" }}>
          <Typography variant="h5" gutterBottom>
            Summary Statistics:
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Statistic</TableCell>
                  <TableCell>Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(response.summary_statistics).map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell>{key}</TableCell>
                    <TableCell>{JSON.stringify(value)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Button
            variant="contained"
            color="secondary"
            disabled={!response}
            sx={{ marginTop: 2 }}
          >
            <a href={`http://127.0.0.1:8000${response.download_link}`} download>
              Download Processed CSV
            </a>
          </Button>
        </div>
      )}
    </div>
  );
}