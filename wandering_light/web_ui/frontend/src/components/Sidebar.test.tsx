import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChakraProvider } from '@chakra-ui/react';
import Sidebar from './Sidebar';
import { FunctionDef } from '../types';

// Wrapper component for Chakra UI
const ChakraWrapper = ({ children }: { children: React.ReactNode }) => (
  <ChakraProvider>{children}</ChakraProvider>
);

const mockFunctions: FunctionDef[] = [
  {
    name: 'add_one',
    input_type: 'builtins.int',
    output_type: 'builtins.int',
    code: 'return x + 1',
    usage_count: 5,
    metadata: {},
  },
  {
    name: 'multiply_two',
    input_type: 'builtins.float',
    output_type: 'builtins.float',
    code: 'return x * 2',
    usage_count: 3,
    metadata: {},
  },
];

const mockOnAddFunction = jest.fn();

describe('Sidebar', () => {
  beforeEach(() => {
    mockOnAddFunction.mockClear();
  });

  it('renders the sidebar with basic structure', () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Node Palette')).toBeInTheDocument();
    expect(screen.getByText('Drag nodes onto the canvas to create your graph')).toBeInTheDocument();
  });

  it('renders TypedList node with default configuration', () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(screen.getByText('TypedList')).toBeInTheDocument();
    expect(screen.getByText('builtins.int')).toBeInTheDocument();
    expect(screen.getAllByText('[1, 2, 3]')[0]).toBeInTheDocument();
  });

  it('renders function nodes from the provided functions list', () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Function Nodes')).toBeInTheDocument();
    expect(screen.getByText('add_one')).toBeInTheDocument();
    expect(screen.getByText('multiply_two')).toBeInTheDocument();
    expect(screen.getByText('Input: builtins.int → Output: builtins.int')).toBeInTheDocument();
    expect(screen.getByText('Input: builtins.float → Output: builtins.float')).toBeInTheDocument();
  });

  it('renders TypedList configuration section', () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Configure TypedList')).toBeInTheDocument();
  });

  it('allows expanding the TypedList configuration', async () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    const configButton = screen.getByText('Configure TypedList');
    await userEvent.click(configButton);

    expect(screen.getByLabelText('Item Type')).toBeInTheDocument();
    expect(screen.getByLabelText('Items (Python list format)')).toBeInTheDocument();
  });

  it('updates TypedList item type when changed', async () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    // Expand configuration
    const configButton = screen.getByText('Configure TypedList');
    await userEvent.click(configButton);

    // Change item type
    const itemTypeInput = screen.getByLabelText('Item Type');
    await userEvent.clear(itemTypeInput);
    await userEvent.type(itemTypeInput, 'builtins.str');

    // Check if the change is reflected in the TypedList display
    expect(screen.getByText('builtins.str')).toBeInTheDocument();
  });

  it('renders add new function form', () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Add New Function')).toBeInTheDocument();
  });

  it('handles empty functions list', () => {
    render(
      <ChakraWrapper>
        <Sidebar functions={[]} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(screen.getByText('Function Nodes')).toBeInTheDocument();
    // Should not show any function nodes
    expect(screen.queryByText('add_one')).not.toBeInTheDocument();
  });

  it('has draggable elements with correct classes', () => {
    const { container } = render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    expect(container.querySelector('.typed-list-node')).toBeInTheDocument();
    expect(container.querySelector('.function-node')).toBeInTheDocument();
  });

  it('handles drag start events', () => {
    const { container } = render(
      <ChakraWrapper>
        <Sidebar functions={mockFunctions} onAddFunction={mockOnAddFunction} />
      </ChakraWrapper>
    );

    // Create a mock drag event
    const mockDataTransfer = {
      setData: jest.fn(),
      effectAllowed: '',
    };

    const typedListNode = container.querySelector('.typed-list-node');
    expect(typedListNode).toBeInTheDocument();

    // Simulate drag start
    fireEvent.dragStart(typedListNode!, {
      dataTransfer: mockDataTransfer,
    });

    expect(mockDataTransfer.setData).toHaveBeenCalledWith('application/reactflow/type', 'typedListNode');
  });
}); 