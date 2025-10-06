import React, { useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  Textarea,
  VStack,
  Heading,
  Text,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Divider,
} from '@chakra-ui/react';
import { FunctionDef } from '../types';

interface SidebarProps {
  functions: FunctionDef[];
  onAddFunction: (fn: FunctionDef) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ functions, onAddFunction }) => {
  const [newFn, setNewFn] = useState<Partial<FunctionDef>>({
    name: '',
    input_type: 'builtins.int',
    output_type: 'builtins.int',
    code: 'return x + 1',
    metadata: {},
  });

  const [typedList, setTypedList] = useState<{ item_type: string, items: string }>({
    item_type: 'builtins.int',
    items: '[1, 2, 3]',
  });

  const parseTypedListItems = (itemsString: string) => {
    try {
      // Try to parse as JSON first
      const parsed = JSON.parse(itemsString);
      if (Array.isArray(parsed)) {
        return parsed;
      }
      throw new Error('Not an array');
    } catch (e) {
      // If JSON parsing fails, try to evaluate as Python-like syntax
      // This is a simple parser for basic cases like [1,2,3] or ['a','b','c']
      const trimmed = itemsString.trim();
      if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
        const inner = trimmed.slice(1, -1);
        if (inner.trim() === '') {
          return [];
        }
        const items = inner.split(',').map(item => {
          const trimmedItem = item.trim();
          // Remove quotes if present
          if ((trimmedItem.startsWith('"') && trimmedItem.endsWith('"')) ||
            (trimmedItem.startsWith("'") && trimmedItem.endsWith("'"))) {
            return trimmedItem.slice(1, -1);
          }
          // Try to parse as number
          const num = parseFloat(trimmedItem);
          if (!isNaN(num)) {
            return Number.isInteger(num) ? parseInt(trimmedItem) : num;
          }
          return trimmedItem;
        });
        return items;
      }
      // Default fallback
      return [itemsString];
    }
  };

  const onDragStart = (event: React.DragEvent<HTMLDivElement>, nodeType: string, data: any) => {
    let nodeData = data;

    // Parse TypedList items when creating the node
    if (nodeType === 'typedListNode' && data.typedList) {
      const parsedItems = parseTypedListItems(data.typedList.items);
      nodeData = {
        ...data,
        typedList: {
          ...data.typedList,
          items: parsedItems
        }
      };
    }

    event.dataTransfer.setData('application/reactflow/type', nodeType);
    event.dataTransfer.setData('application/reactflow/data', JSON.stringify(nodeData));
    event.dataTransfer.effectAllowed = 'move';
  };

  const handleFunctionSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAddFunction(newFn as FunctionDef);
    setNewFn({
      name: '',
      input_type: 'builtins.int',
      output_type: 'builtins.int',
      code: 'return x + 1',
      metadata: {},
    });
  };

  const handleTypedListItemsChange = (value: string) => {
    setTypedList({ ...typedList, items: value });
  };

  return (
    <Box
      w="300px"
      h="100%"
      bg="gray.50"
      p={4}
      borderRight="1px"
      borderColor="gray.200"
      overflowY="auto"
      data-testid="sidebar"
    >
      <VStack spacing={6} align="stretch">
        <Box>
          <Heading size="md" mb={2}>
            Node Palette
          </Heading>
          <Text fontSize="sm" color="gray.600" mb={4}>
            Drag nodes onto the canvas to create your graph
          </Text>
        </Box>

        {/* TypedList Node */}
        <Box
          p={3}
          bg="green.50"
          borderRadius="md"
          border="1px"
          borderColor="green.200"
          className="typed-list-node"
          draggable
          onDragStart={(e) =>
            onDragStart(e, 'typedListNode', {
              label: 'TypedList',
              type: 'typed_list',
              typedList: { ...typedList },
            })
          }
          cursor="grab"
          _hover={{ boxShadow: 'md' }}
        >
          <Heading size="sm" mb={1}>
            TypedList
          </Heading>
          <Text fontSize="xs">{typedList.item_type}</Text>
          <Text fontSize="xs" fontFamily="monospace" mt={1}>
            {(() => {
              try {
                const parsed = parseTypedListItems(typedList.items);
                return `[${parsed.join(', ')}]`;
              } catch (e) {
                return typedList.items;
              }
            })()}
          </Text>
        </Box>

        {/* TypedList Configuration */}
        <Accordion allowToggle>
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left">
                <Text fontWeight="bold">Configure TypedList</Text>
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <VStack spacing={3} align="stretch">
                <FormControl>
                  <FormLabel fontSize="sm">Item Type</FormLabel>
                  <Input
                    size="sm"
                    value={typedList.item_type}
                    onChange={(e) =>
                      setTypedList({ ...typedList, item_type: e.target.value })
                    }
                  />
                </FormControl>
                <FormControl>
                  <FormLabel fontSize="sm">Items (Python list format)</FormLabel>
                  <Textarea
                    size="sm"
                    value={typedList.items}
                    onChange={(e) => handleTypedListItemsChange(e.target.value)}
                    placeholder="[1, 2, 3] or ['a', 'b', 'c']"
                  />
                </FormControl>
              </VStack>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>

        <Divider my={2} />

        {/* Function Nodes from Registry */}
        <Box>
          <Heading size="md" mb={2}>
            Function Nodes
          </Heading>
          <VStack spacing={2} align="stretch">
            {functions.map((fn) => (
              <Box
                key={fn.name}
                p={3}
                bg="blue.50"
                borderRadius="md"
                border="1px"
                borderColor="blue.200"
                className="function-node"
                draggable
                onDragStart={(e) =>
                  onDragStart(e, 'functionNode', {
                    label: fn.name,
                    type: 'function_def',
                    function: fn,
                  })
                }
                cursor="grab"
                _hover={{ boxShadow: 'md' }}
              >
                <Heading size="sm" mb={1}>
                  {fn.name}
                </Heading>
                <Text fontSize="xs">
                  Input: {fn.input_type} → Output: {fn.output_type}
                </Text>
                <Text
                  fontSize="xs"
                  fontFamily="monospace"
                  mt={1}
                  noOfLines={2}
                  title={fn.code}
                >
                  {fn.code}
                </Text>
              </Box>
            ))}
          </VStack>
        </Box>

        <Divider my={2} />

        {/* Add New Function Form */}
        <Accordion allowToggle>
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left">
                <Text fontWeight="bold">Add New Function</Text>
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <form onSubmit={handleFunctionSubmit}>
                <VStack spacing={3} align="stretch">
                  <FormControl isRequired>
                    <FormLabel fontSize="sm">Name</FormLabel>
                    <Input
                      size="sm"
                      value={newFn.name}
                      onChange={(e) => setNewFn({ ...newFn, name: e.target.value })}
                    />
                  </FormControl>
                  <FormControl isRequired>
                    <FormLabel fontSize="sm">Input Type</FormLabel>
                    <Input
                      size="sm"
                      value={newFn.input_type}
                      onChange={(e) => setNewFn({ ...newFn, input_type: e.target.value })}
                    />
                  </FormControl>
                  <FormControl isRequired>
                    <FormLabel fontSize="sm">Output Type</FormLabel>
                    <Input
                      size="sm"
                      value={newFn.output_type}
                      onChange={(e) => setNewFn({ ...newFn, output_type: e.target.value })}
                    />
                  </FormControl>
                  <FormControl isRequired>
                    <FormLabel fontSize="sm">Code</FormLabel>
                    <Textarea
                      size="sm"
                      value={newFn.code}
                      onChange={(e) => setNewFn({ ...newFn, code: e.target.value })}
                      placeholder="return x + 1"
                    />
                  </FormControl>
                  <Button type="submit" colorScheme="blue" size="sm">
                    Add Function
                  </Button>
                </VStack>
              </form>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      </VStack>
    </Box>
  );
};

export default Sidebar;