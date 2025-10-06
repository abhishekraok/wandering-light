import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Text, IconButton } from '@chakra-ui/react';
import { useReactFlow } from 'reactflow';

const FunctionNode = ({ data, selected, id }: NodeProps) => {
  const { label, function: fn, result } = data;
  const { deleteElements } = useReactFlow();

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    deleteElements({ nodes: [{ id }] });
  };

  return (
    <Box className="function-node" boxShadow="md" position="relative">
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#6495ed' }}
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
        {label}
      </Text>

      <Text fontSize="xs" mt={1}>
        {fn.input_type} → {fn.output_type}
      </Text>

      <Text fontSize="xs" fontFamily="monospace" mt={1} noOfLines={1}>
        {fn.code}
      </Text>

      {result && (
        <Box mt={2} p={1} bg="blue.50" borderRadius="sm">
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
        style={{ background: '#6495ed' }}
      />
    </Box>
  );
};

export default memo(FunctionNode);
