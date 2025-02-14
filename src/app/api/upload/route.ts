// app/api/upload/route.ts
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get("file");

  if (!file) {
    return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/process-csv/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Failed to process CSV");
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}