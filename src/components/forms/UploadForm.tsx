// components/FileUploadForm.tsx

import { useState } from 'react';

export default function FileUploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [responseMessage, setResponseMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);

    // Create FormData object to send the file
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send the file to the API endpoint
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        // Trigger download of the processed CSV file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'processed_data.csv'; // Name of the downloaded file
        document.body.appendChild(a);
        a.click();
        a.remove();

        setResponseMessage('File processed and downloaded successfully.');
      } else {
        setResponseMessage('Error processing file.');
      }
    } catch (error) {
      console.error(error);
      setResponseMessage('Network error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Upload CSV File</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Upload'}
        </button>
      </form>

      {responseMessage && <p>{responseMessage}</p>}
    </div>
  );
}