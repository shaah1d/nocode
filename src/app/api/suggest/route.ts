// pages/api/suggest/route.ts
import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai"; 

const geminiKey = process.env.GEMINI_KEY as string;
const genAI = new GoogleGenerativeAI(geminiKey);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

export async function POST(req: NextRequest) {
  try {
    // Parse the request body
    const { question } = await req.json();

    // Construct the prompt
    const prompt = `Here is my problem statement.. make a ML model, I need a paragraph (Strict ML only no DL no AI) with info on the most efficient algorithm and give me the expected range of metrics. Also, put a disclaimer saying that the range of metrics is expected and can vary.
    Make it short to only 4 -5 lines
    Here is the question: ${question}
    dont make a json out of it just do the message in plain message`;

    // Generate content using the Gemini model
    const answer = await model.generateContent(prompt);
    const ans = await answer.response.text();

    // Return the response
    return NextResponse.json({ answer: ans }, { status: 200 });
  } catch (error) {
    // Handle server-side errors
    console.error("Error processing request:", error);
    return NextResponse.json(
      { message: "Internal server error." },
      { status: 500 }
    );
  }
}