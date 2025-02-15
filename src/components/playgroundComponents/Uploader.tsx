// import { useState } from "react";
// import { useRouter } from "next/navigation";
// import { FileUpload } from "@/components/ui/file-upload";
// import {
//   Dialog,
//   DialogContent,
//   DialogDescription,
//   DialogHeader,
//   DialogTitle,
//   DialogTrigger,
// } from "@/components/ui/dialog";
// import { Button } from "@/components/ui/button";
// import { Card, CardContent } from "@/components/ui/card";
// import { Download } from "lucide-react";

// interface ApiResponse {
//   message: string;
//   summary_statistics: Record<string, StatisticsValue>;
//   download_link: string;
// }

// interface StatisticsValue {
//   count: number;
//   mean: number;
//   std: number;
//   min: number;
//   "25%": number;
//   "50%": number;
//   "75%": number;
//   max: number;
// }

// const StatisticsCard = ({ columnName, stats }: { columnName: string; stats: StatisticsValue }) => {
//   const formatValue = (value: number) => {
//     return Number.isInteger(value) ? value.toString() : value.toFixed(4);
//   };

//   const statItems = [
//     { label: "Count", value: stats.count },
//     { label: "Mean", value: stats.mean },
//     { label: "Std Dev", value: stats.std },
//     { label: "Min", value: stats.min },
//     { label: "25%", value: stats["25%"] },
//     { label: "Median", value: stats["50%"] },
//     { label: "75%", value: stats["75%"] },
//     { label: "Max", value: stats.max },
//   ];

//   return (
//     <Card className="w-full">
//       <CardContent className="pt-6">
//         <h3 className="text-lg font-semibold mb-4">{columnName}</h3>
//         <div className="space-y-2">
//           {statItems.map((item) => (
//             <div key={item.label} className="flex justify-between text-sm">
//               <span className="text-gray-600">{item.label}</span>
//               <span className="font-medium">{formatValue(item.value)}</span>
//             </div>
//           ))}
//         </div>
//       </CardContent>
//     </Card>
//   );
// };

// export default function Uploader() {
//   const [file, setFile] = useState<File | null>(null);
//   const [response, setResponse] = useState<ApiResponse | null>(null);
//   const router = useRouter();

//   const handleUpload = async (files: File[]) => {
//     if (files.length === 0) {
//       alert("Please select a file");
//       return;
//     }

//     const selectedFile = files[0];
//     setFile(selectedFile);

//     const formData = new FormData();
//     formData.append("file", selectedFile);

//     try {
//       const res = await fetch("http://127.0.0.1:8000/process-csv/", {
//         method: "POST",
//         body: formData,
//       });
      
//       if (!res.ok) {
//         throw new Error("Failed to process CSV");
//       }
      
//       const data: ApiResponse = await res.json();
//       setResponse(data);
//     } catch (error) {
//       console.error(error);
//       alert("An error occurred while processing the file");
//     }
//   };

//   return (
//     <div className="w-full max-w-4xl mx-auto min-h-96 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg">
//       <FileUpload onChange={handleUpload} />

//       {response && (
//         <Dialog>
//           <DialogTrigger asChild>
//             <Button variant="outline" className="mt-4">
//               Open and download Processed Dataset
//             </Button>
//           </DialogTrigger>
//           <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
//             <DialogHeader>
//               <DialogTitle>Processed Dataset</DialogTitle>
//               <DialogDescription>
//                 Review the summary statistics for each column and download the processed dataset.
//               </DialogDescription>
//             </DialogHeader>
            
//             <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
//               {Object.entries(response.summary_statistics).map(([columnName, stats]) => (
//                 <StatisticsCard key={columnName} columnName={columnName} stats={stats} />
//               ))}
//             </div>

//             <div className="flex justify-end mt-4">
//               <a
//                 href={`http://127.0.0.1:8000${response.download_link}`}
//                 download
//                 className="no-underline"
//               >
//                 <Button className="flex items-center gap-2">
//                   <Download className="w-4 h-4" />
//                   Download Processed CSV
//                 </Button>
//               </a>
//             </div>
//           </DialogContent>
//         </Dialog>
//       )}
//     </div>
//   );
// }


import { useState } from "react";
import { useRouter } from "next/navigation";
import { FileUpload } from "@/components/ui/file-upload";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Download } from "lucide-react";

interface ApiResponse {
  message: string;
  summary_statistics: Record<string, StatisticsValue>;
  download_link: string;
}

// Define a union type for numerical and categorical statistics
interface NumericalStatistics {
  count: number;
  mean: number;
  std: number;
  min: number;
  "25%": number;
  "50%": number;
  "75%": number;
  max: number;
}

interface CategoricalStatistics {
  unique_values: number;
  most_frequent: string;
  frequency_of_most_frequent: number;
  missing_values: number;
}

type StatisticsValue = NumericalStatistics | CategoricalStatistics;

const StatisticsCard = ({ columnName, stats }: { columnName: string; stats: StatisticsValue }) => {
  // Safely format numeric values, handling undefined or null cases
  const formatValue = (value: number | string | undefined) => {
    if (value === undefined || value === null) {
      return "N/A"; // Handle missing values gracefully
    }
    if (typeof value === "number") {
      return Number.isInteger(value) ? value.toString() : value.toFixed(4);
    }
    return value; // Return as-is for strings
  };

  // Check if the stats are for a numerical column
  const isNumerical = "mean" in stats;

  // Define items to display based on the type of statistics
  const statItems = isNumerical
    ? [
        { label: "Count", value: stats.count },
        { label: "Mean", value: stats.mean },
        { label: "Std Dev", value: stats.std },
        { label: "Min", value: stats.min },
        { label: "25%", value: stats["25%"] },
        { label: "Median", value: stats["50%"] },
        { label: "75%", value: stats["75%"] },
        { label: "Max", value: stats.max },
      ]
    : [
        { label: "Unique Values", value: stats.unique_values },
        { label: "Most Frequent", value: stats.most_frequent },
        { label: "Frequency of Most Frequent", value: stats.frequency_of_most_frequent },
        { label: "Missing Values", value: stats.missing_values },
      ];

  return (
    <Card className="w-full">
      <CardContent className="pt-6">
        <h3 className="text-lg font-semibold mb-4">{columnName}</h3>
        <div className="space-y-2">
          {statItems.map((item) => (
            <div key={item.label} className="flex justify-between text-sm">
              <span className="text-gray-600">{item.label}</span>
              <span className="font-medium">{formatValue(item.value)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default function Uploader() {
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const router = useRouter();

  const handleUpload = async (files: File[]) => {
    if (files.length === 0) {
      alert("Please select a file");
      return;
    }

    const selectedFile = files[0];
    setFile(selectedFile);

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

  return (
    <div className="w-full max-w-4xl mx-auto min-h-96 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg">
      <FileUpload onChange={handleUpload} />
      {response && (
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" className="mt-4">
              Open and download Processed Dataset
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Processed Dataset</DialogTitle>
              <DialogDescription>
                Review the summary statistics for each column and download the processed dataset.
              </DialogDescription>
            </DialogHeader>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
              {Object.entries(response.summary_statistics).map(([columnName, stats]) => (
                <StatisticsCard key={columnName} columnName={columnName} stats={stats} />
              ))}
            </div>
            <div className="flex justify-end mt-4">
              <a
                href={`http://127.0.0.1:8000${response.download_link}`}
                download
                className="no-underline"
              >
                <Button className="flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Download Processed CSV
                </Button>
              </a>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}