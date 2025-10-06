import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  FormControl,
  FormLabel,
  Heading,
  Input,
  VStack,
  HStack,
  Text,
  Tag,
  TagLabel,
  Wrap,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Divider,
  Code,
} from '@chakra-ui/react';
import axios from 'axios';

interface TypedListData {
  item_type: string;
  items: any[];
}

interface SolverResponse {
  success: boolean;
  predicted_functions: string[];
  predicted_output: TypedListData | null;
  error_msg: string | null;
}

const SolverPlayground: React.FC = () => {
  const [inputType, setInputType] = useState('builtins.int');
  const [inputItems, setInputItems] = useState('[1, 2, 3]');
  const [outputType, setOutputType] = useState('builtins.int');
  const [outputItems, setOutputItems] = useState('[2, 4, 6]');
  const [checkpointPath, setCheckpointPath] = useState('../../../checkpoints/saved/rl/long_sft_opt_125m_s35k_no_len/');
  const [solverResult, setSolverResult] = useState<SolverResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const handleSubmit = async () => {
    setIsLoading(true);
    setSolverResult(null);

    try {
      // Parse input and output items
      const inputItemsParsed = JSON.parse(inputItems);
      const outputItemsParsed = JSON.parse(outputItems);

      const requestData = {
        input_list: {
          item_type: inputType,
          items: inputItemsParsed,
        },
        output_list: {
          item_type: outputType,
          items: outputItemsParsed,
        },
        checkpoint_path: checkpointPath,
      };

      const response = await axios.post<SolverResponse>(
        'http://localhost:8000/solver/execute',
        requestData
      );

      setSolverResult(response.data);
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to execute solver',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const formatTypedList = (typedList: TypedListData) => {
    return `${typedList.item_type}: ${JSON.stringify(typedList.items)}`;
  };

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8} align="stretch">
        <Heading>Solver Playground</Heading>
        
        <Box borderWidth={1} borderRadius="lg" p={6}>
          <VStack spacing={4} align="stretch">
            <Heading size="md">Problem Definition</Heading>
            <Text fontSize="sm" color="gray.600">
              Define the input and the expected output. The solver will try to find functions that transform the input into the expected output.
            </Text>
            
            <HStack spacing={4}>
              <FormControl flex={1}>
                <FormLabel>Input Type</FormLabel>
                <Input
                  value={inputType}
                  onChange={(e) => setInputType(e.target.value)}
                  placeholder="e.g., builtins.int"
                />
              </FormControl>
              
              <FormControl flex={2}>
                <FormLabel>Input Items (JSON array)</FormLabel>
                <Input
                  value={inputItems}
                  onChange={(e) => setInputItems(e.target.value)}
                  placeholder="e.g., [1, 2, 3]"
                />
              </FormControl>
            </HStack>
            
            <HStack spacing={4}>
              <FormControl flex={1}>
                <FormLabel>Expected Output Type</FormLabel>
                <Input
                  value={outputType}
                  onChange={(e) => setOutputType(e.target.value)}
                  placeholder="e.g., builtins.int"
                />
              </FormControl>
              
              <FormControl flex={2}>
                <FormLabel>Expected Output Items (JSON array)</FormLabel>
                <Input
                  value={outputItems}
                  onChange={(e) => setOutputItems(e.target.value)}
                  placeholder="e.g., [2, 4, 6]"
                />
              </FormControl>
            </HStack>
          </VStack>
        </Box>
        
        <Box borderWidth={1} borderRadius="lg" p={6}>
          <VStack spacing={4} align="stretch">
            <Heading size="md">Model Configuration</Heading>
            
            <FormControl>
              <FormLabel>Checkpoint Path</FormLabel>
              <Input
                value={checkpointPath}
                onChange={(e) => setCheckpointPath(e.target.value)}
                placeholder="Path to model checkpoint"
              />
            </FormControl>
          </VStack>
        </Box>
        
        <Button
          colorScheme="blue"
          size="lg"
          onClick={handleSubmit}
          isLoading={isLoading}
          loadingText="Running Solver..."
        >
          Run Solver
        </Button>
        
        {solverResult && (
          <Box borderWidth={1} borderRadius="lg" p={6}>
            <VStack spacing={4} align="stretch">
              <Heading size="md">Results</Heading>
              
              {solverResult.success ? (
                <Alert status="success">
                  <AlertIcon />
                  <AlertTitle>Success!</AlertTitle>
                  <AlertDescription>
                    The solver found a valid trajectory.
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert status="error">
                  <AlertIcon />
                  <AlertTitle>Failed</AlertTitle>
                  <AlertDescription>
                    {solverResult.error_msg || 'The solver could not find a valid trajectory.'}
                  </AlertDescription>
                </Alert>
              )}
              
              <Divider />
              
              <Box>
                <Text fontWeight="bold" mb={2}>Solver's Predicted Functions (to achieve expected output):</Text>
                {solverResult.predicted_functions.length > 0 ? (
                  <Wrap>
                    {solverResult.predicted_functions.map((fn, idx) => (
                      <Tag key={idx} size="lg" colorScheme="green">
                        <TagLabel>{fn}</TagLabel>
                      </Tag>
                    ))}
                  </Wrap>
                ) : (
                  <Text color="gray.500">No functions predicted</Text>
                )}
              </Box>
              
              {solverResult.predicted_output && (
                <Box>
                  <Text fontWeight="bold" mb={2}>Solver's Predicted Output:</Text>
                  <Code p={2} display="block">
                    {formatTypedList(solverResult.predicted_output)}
                  </Code>
                </Box>
              )}
            </VStack>
          </Box>
        )}
      </VStack>
    </Container>
  );
};

export default SolverPlayground;
