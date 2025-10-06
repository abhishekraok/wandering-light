import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Text, IconButton } from '@chakra-ui/react';
import { useReactFlow } from 'reactflow';

const TypedListNode = ({ data, selected, id }: NodeProps) => {
  const { label, typedList, result } = data;
  const { deleteElements } = useReactFlow();

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <Box className="typed-list-node" boxShadow="md" position="relative">
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#90ee90' }}
      />

      {(selected) && (
        <IconButton
          aria-label="Delete node"
          icon={<Text fontSize="xs">×</Text>}
          size="xs"
          position="absolute"
          top="-8px"
          right="-8px"
          borderRadius="full"
          bg="red.500"
          color="white"
          _hover={{ bg: 'red.600' }}
          onClick={handleDelete}
          zIndex={10}
        />
      )}

      <Text fontWeight="bold" fontSize="sm">
        {label || 'TypedList'}
      </Text>

      <Text fontSize="xs" mt={1}>
        {typedList.item_type}
      </Text>

      <Text fontSize="xs" fontFamily="monospace" mt={1} noOfLines={1}>
        {Array.isArray(typedList.items)
          ? `[${typedList.items.join(', ')}]`
          : typedList.items
        }
      </Text>

      {result && (
        <Box mt={2} p={1} bg="green.50" borderRadius="sm">
          <Text fontSize="xs" fontWeight="bold">
            Result:
          </Text>
          <Text fontSize="xs" fontFamily="monospace">
            [{result.join(', ')}]
          </Text>
        </Box>
      )}

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#90ee90' }}
      />
    </Box>
  );
};

export default memo(TypedListNode);
