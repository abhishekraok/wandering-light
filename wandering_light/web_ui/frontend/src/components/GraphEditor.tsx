import React, { useState, useRef, useCallback, useEffect } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Connection,
  Background,
  ConnectionLineType,
  NodeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { Box, Button, Flex, useToast } from '@chakra-ui/react';
import { FunctionDef, ExecutionResult } from '../types';
import Sidebar from './Sidebar';
import FunctionNode from './nodes/FunctionNode';
import TypedListNode from './nodes/TypedListNode';
import axios from 'axios';

// Define custom node types
const nodeTypes: NodeTypes = {
  functionNode: FunctionNode,
  typedListNode: TypedListNode,
};

// API base URL
const API_URL = 'http://localhost:8000';

const GraphEditor: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [functions, setFunctions] = useState<FunctionDef[]>([]);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [isAutoExecuting, setIsAutoExecuting] = useState(false);
  // Create a toast notification helper
  const toast = useToast();

  // Load functions from the API
  useEffect(() => {
    const fetchFunctions = async () => {
      try {
        const response = await axios.get(`${API_URL}/functions`);
        setFunctions(response.data);
      } catch (error) {
        console.error('Error fetching functions:', error);
        toast({
          title: 'Error',
          description: 'Failed to load functions from API',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };

    fetchFunctions();
  }, [toast]);

  // Setup WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/execute-graph`);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setWebsocket(ws);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.status === 'success') {
        // Update nodes with results
        setNodes((nds) =>
          nds.map((node) => {
            const result = data.results.find(
              (r: ExecutionResult) => r.node_id === node.id
            );

            if (result) {
              return {
                ...node,
                data: {
                  ...node.data,
                  result: result.result,
                },
              };
            }
            return node;
          })
        );

        setIsAutoExecuting(false);
        toast({
          title: 'Execution Complete',
          description: 'Graph executed successfully',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      } else if (data.status === 'error') {
        setIsAutoExecuting(false);
        toast({
          title: 'Execution Error',
          description: data.error,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      toast({
        title: 'WebSocket Error',
        description: 'Failed to connect to server',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setWebsocket(null);
    };

    return () => {
      ws.close();
    };
  }, [setNodes, toast, setIsAutoExecuting]);

  // Handle new connections
  const onConnect = useCallback(
    (params: Connection) => {
      // Add new edge
      const newEdge = {
        ...params,
        type: 'smoothstep',
        animated: true
      };

      setEdges((eds) => {
        const updatedEdges = addEdge(newEdge, eds);

        // Trigger automatic execution after edge is added
        setTimeout(() => {
          if (websocket && websocket.readyState === WebSocket.OPEN) {
            setIsAutoExecuting(true);

            // Format and send graph data with the new edge
            const graphData = {
              nodes: nodes.map(node => ({
                id: node.id,
                type: node.data.type,
                data: node.data.type === 'function_def'
                  ? { name: node.data.function.name }
                  : {
                    item_type: node.data.typedList.item_type,
                    items: node.data.typedList.items
                  }
              })),
              edges: updatedEdges
            };

            websocket.send(JSON.stringify(graphData));

            toast({
              title: 'Auto-executing',
              description: 'Running connected components...',
              status: 'info',
              duration: 2000,
              isClosable: true,
            });
          }
        }, 100); // Small delay to ensure state is updated

        return updatedEdges;
      });
    },
    [setEdges, nodes, websocket, toast, setIsAutoExecuting]
  );

  // Handle drag over for new nodes
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle dropping new nodes
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow/type');
      const nodeData = JSON.parse(event.dataTransfer.getData('application/reactflow/data'));

      // Get position from drop coordinates
      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      // Create a new node
      const newNode = {
        id: `${type}_${Date.now()}`,
        type,
        position,
        data: { ...nodeData },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  // Execute the graph
  const executeGraph = useCallback(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      setIsAutoExecuting(true);

      // Format and send graph data
      const graphData = {
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.data.type,
          data: node.data.type === 'function_def'
            ? { name: node.data.function.name }
            : {
              item_type: node.data.typedList.item_type,
              items: node.data.typedList.items
            }
        })),
        edges
      };

      websocket.send(JSON.stringify(graphData));
    } else {
      toast({
        title: 'Connection Error',
        description: 'WebSocket not connected',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  }, [nodes, edges, websocket, toast, setIsAutoExecuting]);

  return (
    <ReactFlowProvider>
      <Flex h="100%" w="100%" data-testid="graph-editor">
        <Sidebar
          functions={functions}
          onAddFunction={async (fn: FunctionDef) => {
            try {
              const response = await axios.post(`${API_URL}/functions`, fn);
              setFunctions(prev => [...prev, response.data]);
              toast({
                title: 'Success',
                description: `Function "${fn.name}" added`,
                status: 'success',
                duration: 3000,
                isClosable: true,
              });
            } catch (error) {
              console.error('Error adding function:', error);
              toast({
                title: 'Error',
                description: 'Failed to add function',
                status: 'error',
                duration: 5000,
                isClosable: true,
              });
            }
          }}
        />
        <Box flex="1" ref={reactFlowWrapper} h="100%" data-testid="graph-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            connectionLineType={ConnectionLineType.SmoothStep}
            deleteKeyCode={['Delete', 'Backspace']}
            multiSelectionKeyCode={['Meta', 'Control']}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
          <Box position="absolute" bottom="20px" right="20px" zIndex={10}>
            <Button
              colorScheme="green"
              onClick={executeGraph}
              size="lg"
              isDisabled={!websocket || nodes.length === 0 || isAutoExecuting}
              isLoading={isAutoExecuting}
              loadingText="Auto-executing..."
              data-testid="execute-graph-button"
            >
              Execute Graph
            </Button>
          </Box>
        </Box>
      </Flex>
    </ReactFlowProvider>
  );
};

export default GraphEditor;