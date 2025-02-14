"use client"
import React, { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import Uploader from './Uploader';
import Train from './Train';
import { ScrollArea } from '@radix-ui/react-scroll-area';
import Hyper from './Hyper';
import Testing from './Testing';
import { X } from 'lucide-react';
import DataVisualization from './DataVisualization';


interface SidebarCard {
  id: string;
  title: string;
}

interface DropZone {
  id: string;
  title: string;
  position: { x: number; y: number };
  occupied: string | null;
}

const FlowchartComponent: React.FC = () => {
  const [sidebarCards, setSidebarCards] = useState<SidebarCard[]>([
    { id: 'card1', title: 'Import dataset & Data cleaning' },
    { id: 'card2', title: 'Train model' },
    { id: 'card3', title: 'Hyper-Parameter Tuning' },
    { id: 'card4', title: 'Extensive Model Testing' },
    { id: 'card5', title: 'Model Data Visualization' },
  ]);

  // Adjusted positions for center alignment
  const [dropZones] = useState<DropZone[]>([
    { id: 'start', title: 'Start', position: { x: 0, y: 0 }, occupied: null },
    { id: 'process1', title: 'Process 1', position: { x: 0, y: 100 }, occupied: null },
    { id: 'decision', title: 'Decision', position: { x: 0, y: 200 }, occupied: null },
    { id: 'process2', title: 'Process 2', position: { x: 0, y: 300 }, occupied: null },
    { id: 'end', title: 'End', position: { x: 0, y: 400 }, occupied: null },
  ]);

  const [occupiedZones, setOccupiedZones] = useState<Record<string, string>>({});
  const [draggedItem, setDraggedItem] = useState<SidebarCard | null>(null);
  const [openData1, setOpenData1] = useState(false);
  const [openData2, setOpenData2] = useState(false);
  const [openData3, setOpenData3] = useState(false);
  const [openData4, setOpenData4] = useState(false);
  const [openData5, setOpenData5] = useState(false);

  useEffect(() => {
    if (occupiedZones['start'] === 'card1') setOpenData1(true);
    else setOpenData1(false);
    if (occupiedZones['process1'] === 'card2') setOpenData2(true);
    else setOpenData2(false);
    if (occupiedZones['decision'] === 'card3') setOpenData3(true);
    else setOpenData3(false);
    if (occupiedZones['process2'] === 'card4') setOpenData4(true);
    else setOpenData4(false);
    if (occupiedZones['end'] === 'card5') setOpenData5(true);
    else setOpenData5(false);
  }, [occupiedZones]);

  const handleDrop = (e: React.DragEvent, zoneId: string) => {
    e.preventDefault();
    if (!draggedItem) return;
    setOccupiedZones((prev) => ({
      ...prev,
      [zoneId]: draggedItem.id,
    }));
    setSidebarCards((prev) => prev.filter((card) => card.id !== draggedItem.id));
    setDraggedItem(null);
  };

  const handleRemove = (zoneId: string, cardId: string) => {
    setOccupiedZones((prev) => {
      const newOccupied = { ...prev };
      delete newOccupied[zoneId];
      return newOccupied;
    });

    const cardToAdd = {
      id: cardId,
      title: sidebarCards.find((card) => card.id === cardId)?.title ||
        (cardId === 'card1' ? 'Import dataset & Data cleaning' :
          cardId === 'card2' ? 'Train model' :
            cardId === 'card3' ? 'Hyper-Parameter Tuning' :
              cardId === 'card4' ? 'Extensive Model Testing' :
                'Model Data Visualization')
    };

    setSidebarCards((prev) => [...prev, cardToAdd]);
  };

  return (
    <div className="flex h-screen w-full bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-white p-4 shadow-lg">
        <h2 className="text-lg font-semibold mb-4">Components</h2>
        {sidebarCards.map((card) => (
          <Card
            key={card.id}
            className={`mb-4 p-4 cursor-move hover:shadow-lg transition-shadow ${
              card.id === 'card1' ? 'bg-blue-200' :
              card.id === 'card2' ? 'bg-green-200' :
              card.id === 'card3' ? 'bg-yellow-200' :
              card.id === 'card4' ? 'bg-red-200' :
              'bg-purple-200'
            }`}
            draggable
            onDragStart={(e) => setDraggedItem(card)}
          >
            {card.title}
          </Card>
        ))}
      </div>

      {/* Main Canvas with Dot Grid */}
      <div className="flex-1 relative overflow-auto">
        {/* Dot Grid Background */}
    
        
        {/* Centered Flowchart Container */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative" style={{ width: '200px', height: '500px' }}>
            {dropZones.map((zone) => (
              <div
                key={zone.id}
                className="absolute w-48 h-24 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center"
                style={{
                  left: '50%',
                  transform: 'translateX(-50%)',
                  top: zone.position.y,
                  backgroundColor: occupiedZones[zone.id] ? 'white' : 'transparent',
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => handleDrop(e, zone.id)}
              >
                {occupiedZones[zone.id] ? (
                  <div className={`w-full h-full p-4 relative ${
                    occupiedZones[zone.id] === 'card1' ? 'bg-blue-200' :
                    occupiedZones[zone.id] === 'card2' ? 'bg-green-200' :
                    occupiedZones[zone.id] === 'card3' ? 'bg-yellow-200' :
                    occupiedZones[zone.id] === 'card4' ? 'bg-red-200' :
                    'bg-purple-200'
                  }`}>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">{
                        occupiedZones[zone.id] === 'card1' ? 'Import dataset & Data cleaning' :
                        occupiedZones[zone.id] === 'card2' ? 'Train model' :
                        occupiedZones[zone.id] === 'card3' ? 'Hyper-Parameter Tuning' :
                        occupiedZones[zone.id] === 'card4' ? 'Extensive Model Testing' :
                        'Model Data Visualization'
                      }</span>
                      <button
                        onClick={() => handleRemove(zone.id, occupiedZones[zone.id]!)}
                        className="absolute top-2 right-2 p-1 hover:bg-gray-100 rounded-full"
                      >
                        <X size={16} className="text-gray-500" />
                      </button>
                    </div>
                  </div>
                ) : (
                  <span className="text-gray-400">{zone.title}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Component */}
      <div className="w-96 bg-white p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Details</h2>
        <ScrollArea className="h-[calc(100vh-100px)]">
          {openData5 && <DataVisualization />}
          {!openData5 && openData4 && <Testing />}
          {!openData5 && !openData4 && openData3 && <Hyper />}
          {!openData5 && !openData4 && !openData3 && openData2 && <Train />}
          {!openData5 && !openData4 && !openData3 && !openData2 && openData1 && <Uploader />}
          {!openData1 && !openData2 && !openData3 && !openData4 && !openData5 && (
            <div className="text-gray-500 text-center">
              <p>No component selected</p>
              <p>Drag and drop a component to view details</p>
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
};

export default FlowchartComponent;