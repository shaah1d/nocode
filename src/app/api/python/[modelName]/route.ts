// app/api/python/[modelName]/route.ts
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: Request, { params }: { params: { modelName: string } }) {
  try {
    // Extract the modelName from the dynamic route parameter
    const { modelName } = await params;

    // Construct the path to the Python script dynamically
    const pythonScriptPath = path.join(process.cwd(), `pymodules/${modelName}.py`);

    // Read the file synchronously
    const fileContent = fs.readFileSync(pythonScriptPath, 'utf-8');

    // Return the file content as a JSON response
    return NextResponse.json({ content: fileContent });
  } catch (error) {
    console.error('Error reading Python file:', error);
    return NextResponse.json(
      { error: 'Failed to read Python file' },
      { status: 500 }
    );
  }
}