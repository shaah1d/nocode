// app/api/download/route.ts
import { NextResponse } from "next/server";
import { join } from "path";
import { existsSync } from "fs";

export async function GET(request: Request, { params }: { params: { file_name: string } }) {
  const fileName = params.file_name;
  const filePath = join(process.cwd(), "temp", fileName);

  if (!existsSync(filePath)) {
    return NextResponse.json({ error: "File not found" }, { status: 404 });
  }

  return new NextResponse(Bun.file(filePath), {
    headers: {
      "Content-Disposition": `attachment; filename=${fileName}`,
      "Content-Type": "text/csv",
    },
  });
}