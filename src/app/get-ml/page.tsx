// app/page.tsx
"use client";
import { useState } from "react";

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch('/api/gemini', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) {
        throw new Error('Failed to fetch data from Gemini API');
      }

      const data = await res.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
      alert('An error occurred while processing your request');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: "2rem" }}>
      <h1>AI/ML Model Suggestion</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Describe your use case:
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your use case here..."
            style={{ width: "100%", height: 100, marginTop: 10 }}
          />
        </label>
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Loading..." : "Get Suggestions"}
        </button>
      </form>

      {response && (
        <div style={{ marginTop: 20 }}>
          <h2>Suggestions:</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}