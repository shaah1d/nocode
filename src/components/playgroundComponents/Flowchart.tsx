"use client"
import React, { useCallback, useState } from "react";
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Background,
  Controls,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Uploader from "./Uploader";
import Train from "./Train";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import Hyper from "./Hyper";
import Testing from "./Testing";
import DataVisualization from "./DataVisualization";



export default function FlowchartComponent() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [openData1, setOpenData1] = useState(false);
  const [openData2, setOpenData2] = useState(false);
  const [openData3, setOpenData3] = useState(false);
  const [openData4, setOpenData4] = useState(false);
  const [openData5, setOpenData5] = useState(false);

  // Callback for connecting edges
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge(params, eds));
      if (params.source === "1" || params.target === "1") setOpenData1(true);
      if (params.source === "2" || params.target === "2") setOpenData2(true);
      if (params.source === "3" || params.target === "3") setOpenData3(true);
      if (params.source === "4" || params.target === "4") setOpenData4(true);
      if (params.source === "5" || params.target === "5") setOpenData5(true);
    },
    [setEdges]
  );

  // Handle drag start for nodes in the left sidebar
  const onDragStart = (event: React.DragEvent<HTMLDivElement>, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";

   
  };

  // Handle drop on the ReactFlow canvas
  // Handle drop on the ReactFlow canvas
const onDrop = useCallback(
  (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    // Get the node type from the drag data
    const type = event.dataTransfer.getData("application/reactflow");
    if (!type) return;

    // Get the mouse position relative to the ReactFlow canvas
    const reactFlowBounds = event.currentTarget.getBoundingClientRect();
    const position = {
      x: event.clientX - reactFlowBounds.left,
      y: event.clientY - reactFlowBounds.top,
    };

    // Add a new node to the state
    const newNode = {
      id: `${nodes.length + 1}`,
      type,
      position,
      data: { label: `${type} ` },
      style: { backgroundColor: getRandomColor() },
    };
    setNodes((nds) => nds.concat(newNode));

    // Check if the dropped node is "Card 1" and set openData1 to true
    if (type === "Import Dataset & Data Cleaning") {
      setOpenData1(true);
    }
  },
  [nodes, setNodes]
);

  // Allow dropping on the ReactFlow canvas
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  return (
<div style={{ display: "flex", width: "100vw", height: "100vh", overflow: "hidden" }}>
  {/* Left Section: Node Templates */}
  <div className="w-64 bg-gray-100 p-4 shadow-lg overflow-y-auto">
    <h2 className="text-xl font-bold mb-4 text-gray-800">Node Templates</h2>
    <div
      className="p-4 mb-4 bg-purple-200 rounded-lg shadow-md cursor-move transition duration-300 ease-in-out hover:bg-purple-300 hover:shadow-lg"
      onDragStart={(event) => onDragStart(event, "Import Dataset & Data Cleaning")}
      draggable
    >
      Import Dataset & Data Cleaning
    </div>
    <div
      className="p-4 mb-4 bg-blue-200 rounded-lg shadow-md cursor-move transition duration-300 ease-in-out hover:bg-blue-300 hover:shadow-lg"
      onDragStart={(event) => onDragStart(event, "Train Model")}
      draggable
    >
      Train Model
    </div>
    <div
      className="p-4 mb-4 bg-green-200 rounded-lg shadow-md cursor-move transition duration-300 ease-in-out hover:bg-green-300 hover:shadow-lg"
      onDragStart={(event) => onDragStart(event, "Hyper-Parameter Tuning")}
      draggable
    >
      Hyper-Parameter Tuning
    </div>
    <div
      className="p-4 mb-4 bg-orange-200 rounded-lg shadow-md cursor-move transition duration-300 ease-in-out hover:bg-orange-300 hover:shadow-lg"
      onDragStart={(event) => onDragStart(event, "Extensive Model Testing")}
      draggable
    >
      Extensive Model Testing
    </div>
    <div
      className="p-4 mb-4 bg-red-200 rounded-lg shadow-md cursor-move transition duration-300 ease-in-out hover:bg-red-300 hover:shadow-lg"
      onDragStart={(event) => onDragStart(event, "Data Visualization")}
      draggable
    >
      Data Visualization
    </div>
  </div>

  {/* Center Section: ReactFlow Canvas */}
  <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onDrop={onDrop}
      onDragOver={onDragOver}
      style={{ width: "100%", height: "100%" }}
    >
      <Background />
      <Controls />
    </ReactFlow>
  </div>

  {/* Right Section: Details */}
  <div className="w-96 bg-white p-6 shadow-lg overflow-y-auto">
    <h2 className="text-2xl font-bold mb-6 text-gray-800">Details</h2>
    <ScrollArea className="h-[calc(100vh-150px)]">
      {openData5 && <DataVisualization />}
      {!openData5 && openData4 && <Testing />}
      {!openData5 && !openData4 && openData3 && <Hyper />}
      {!openData5 && !openData4 && !openData3 && openData2 && <Train />}
      {!openData5 && !openData4 && !openData3 && !openData2 && openData1 && (
        <Uploader />
      )}
      {!openData1 &&
        !openData2 &&
        !openData3 &&
        !openData4 &&
        !openData5 && (
          <div className="text-gray-500 text-center">
            <p>No component selected</p>
            <p>Drag and drop a component to view details</p>
          </div>
        )}
    </ScrollArea>
  </div>
</div>
  );
}

// Helper function to generate random colors for nodes
function getRandomColor() {
  const colors = ["pink", "lightblue", "lightgreen", "orange", "red"];
  return colors[Math.floor(Math.random() * colors.length)];
}