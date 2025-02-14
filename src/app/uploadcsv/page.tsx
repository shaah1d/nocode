// app/page.tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation"; // Import useRouter for navigation
import { Button, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import FileUploadIcon from '@mui/icons-material/FileUpload';

interface ApiResponse {
  message: string;
  summary_statistics: Record<string, any>;
  download_link: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const router = useRouter(); // Initialize useRouter

  const handleUpload = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

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
    // Navigate to the /readpy route
    router.push("/readpy");
  };

  return (
    <div style={{ maxWidth: 800, margin: "auto", padding: "2rem" }}>
      <Paper elevation={3} style={{ padding: "2rem", textAlign: "center" }}>
        <Typography variant="h4" gutterBottom>
          Upload CSV for Processing
        </Typography>

        {/* File Upload Form */}
        <form onSubmit={handleUpload}>
          <label htmlFor="file-upload" style={{ cursor: "pointer" }}>
            <input
              id="file-upload"
              type="file"
              accept=".csv"
              hidden
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <Button
              variant="contained"
              component="span"
              startIcon={<FileUploadIcon />}
              sx={{ marginBottom: 2 }}
            >
              Select CSV File
            </Button>
          </label>
          <br />
          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={!file}
            sx={{ marginTop: 2 }}
          >
            Upload
          </Button>
        </form>

        {/* Display Summary Statistics */}
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

            {/* Download Link */}
            <Typography variant="h6" style={{ marginTop: "2rem" }}>
              <a href={`http://127.0.0.1:8000${response.download_link}`} download>
                Download Processed CSV
              </a>
            </Typography>

            {/* Next Step Button */}
            <Button
              variant="contained"
              color="secondary"
              onClick={handleNextStep}
              disabled={!response} // Enable only if a response exists
              sx={{ marginTop: 2 }}
            >
              Next Step
            </Button>
          </div>
        )}
      </Paper>
    </div>
  );
}