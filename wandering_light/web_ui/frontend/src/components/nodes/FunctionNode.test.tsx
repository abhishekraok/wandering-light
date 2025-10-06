import React from 'react';
import { render, screen } from '@testing-library/react';
import { ChakraProvider } from '@chakra-ui/react';
import FunctionNode from './FunctionNode';
import { FunctionDef } from '../../types';

// Mock the reactflow exports used in this component so that we don't need the
// ReactFlow provider in tests.
jest.mock('reactflow', () => ({
  __esModule: true,
  Handle: ({ children, type }: any) => (
    <div data-testid={`handle-${type}`}>{children}</div>
  ),
  Position: { Left: 'left', Right: 'right' },
  useReactFlow: () => ({ deleteElements: jest.fn() }),
}));

// Wrapper component for Chakra UI
const ChakraWrapper = ({ children }: { children: React.ReactNode }) => (
  <ChakraProvider>{children}</ChakraProvider>
);

const mockFunction: FunctionDef = {
  name: 'add_one',
  input_type: 'builtins.int',
  output_type: 'builtins.int',
  code: 'return x + 1',
  usage_count: 5,
  metadata: {},
};

// Test helper to create valid props
const createTestProps = (overrides: any = {}) => ({
  id: 'test-node',
  type: 'functionNode',
  data: {
    label: 'Add One',
    type: 'function_def' as const,
    function: mockFunction,
    ...overrides,
  },
  selected: false,
  zIndex: 1,
  isConnectable: true,
  xPos: 0,
  yPos: 0,
  dragging: false,
});

describe('FunctionNode', () => {
  it('renders function node with basic information', () => {
    const props = createTestProps();
    render(
      <ChakraWrapper>
        <FunctionNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Add One')).toBeInTheDocument();
    expect(screen.getByText('builtins.int → builtins.int')).toBeInTheDocument();
    expect(screen.getByText('return x + 1')).toBeInTheDocument();
  });

  it('renders function node with result when result is provided', () => {
    const props = createTestProps({ result: [2, 3, 4] });
    render(
      <ChakraWrapper>
        <FunctionNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Add One')).toBeInTheDocument();
    expect(screen.getByText('Result:')).toBeInTheDocument();
    expect(screen.getByText('[2, 3, 4]')).toBeInTheDocument();
  });

  it('does not render result section when no result is provided', () => {
    const props = createTestProps();
    render(
      <ChakraWrapper>
        <FunctionNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.queryByText('Result:')).not.toBeInTheDocument();
  });

  it('displays function name and types correctly', () => {
    const customFunction: FunctionDef = {
      name: 'multiply',
      input_type: 'builtins.float',
      output_type: 'builtins.str',
      code: 'return str(x * 2)',
      usage_count: 10,
      metadata: {},
    };

    const props = createTestProps({
      label: 'Multiply Function',
      function: customFunction,
    });

    render(
      <ChakraWrapper>
        <FunctionNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Multiply Function')).toBeInTheDocument();
    expect(screen.getByText('builtins.float → builtins.str')).toBeInTheDocument();
    expect(screen.getByText('return str(x * 2)')).toBeInTheDocument();
  });

  it('renders with function node class', () => {
    const props = createTestProps();
    const { container } = render(
      <ChakraWrapper>
        <FunctionNode {...props} />
      </ChakraWrapper>
    );

    expect(container.querySelector('.function-node')).toBeInTheDocument();
  });

  it('renders source and target handles', () => {
    const props = createTestProps();
    render(
      <ChakraWrapper>
        <FunctionNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByTestId('handle-target')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source')).toBeInTheDocument();
  });
});
