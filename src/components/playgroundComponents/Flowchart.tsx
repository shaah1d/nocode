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
  MiniMap,
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
  const [availableNodes, setAvailableNodes] = useState([
    { type: "Import Dataset & Data Cleaning", color: "bg-purple-200", hoverColor: "bg-purple-300" },
    { type: "Train Model", color: "bg-blue-200", hoverColor: "bg-blue-300" },
    { type: "Hyper-Parameter Tuning", color: "bg-green-200", hoverColor: "bg-green-300" },
    { type: "Extensive Model Testing", color: "bg-orange-200", hoverColor: "bg-orange-300" },
    { type: "Data Visualization", color: "bg-red-200", hoverColor: "bg-red-300" },
  ]);

 // This is to match the colors from the cards to the nodes 
  const nodeColorMap = {
    "Import Dataset & Data Cleaning": "#e9d5ff", // purple-200
    "Train Model": "#bfdbfe", // blue-200
    "Hyper-Parameter Tuning": "#bbf7d0", // green-200
    "Extensive Model Testing": "#fed7aa", // orange-200
    "Data Visualization": "#fecaca", // red-200
  };

  // This function is to render steps based on connections of edges using states
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

  
  const onDragStart = (event: React.DragEvent<HTMLDivElement>, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType);
    event.dataTransfer.effectAllowed = "move";
  };

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

     
      const type = event.dataTransfer.getData("application/reactflow");
      if (!type) return;

     
      const reactFlowBounds = event.currentTarget.getBoundingClientRect();
      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      // Here once card is dropped a new node is created with the color given above
      const newNode = {
        id: `${nodes.length + 1}`,
        type,
        position,
        data: { label: `${type} ` },
        style: { backgroundColor: nodeColorMap[type] },
      };
      setNodes((nds) => nds.concat(newNode));

      // Remove the node from available nodes
      setAvailableNodes(availableNodes.filter(node => node.type !== type));

      if (type === "Import Dataset & Data Cleaning") {
        setOpenData1(true);
      }
    },
    [nodes, setNodes, availableNodes, nodeColorMap]
  );

  // Allow dropping on the ReactFlow canvas
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  return (
    <div style={{ display: "flex", width: "98.5vw", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar with cards */}
      <div className="w-64 bg-gray-100 p-4 shadow-lg overflow-y-auto">
        <h2 className="text-xl font-bold mb-4 text-gray-800">Node Templates</h2>
        {availableNodes.map((node, index) => (
          <div
            key={index}
            className={`p-4 mb-4 ${node.color} rounded-lg shadow-md cursor-move transition duration-300 ease-in-out hover:${node.hoverColor} hover:shadow-lg`}
            onDragStart={(event) => onDragStart(event, node.type)}
            draggable
          >
            {node.type}
          </div>
        ))}
        {availableNodes.length === 0 && (
          <div className="text-gray-500 text-center p-4">
            <p>All components have been used</p>
          </div>
        )}
      </div>

      {/*  ReactFlow Canvas */}
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
          {/* <MiniMap /> */}
        </ReactFlow>
      </div>

      {/* conditional rendered pages */}
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