import React from 'react';
import { render, screen } from '@testing-library/react';
import { ChakraProvider } from '@chakra-ui/react';
import TypedListNode from './TypedListNode';

// Mock reactflow components used by TypedListNode so tests don't require the
// real provider from ReactFlow.
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

// Test helper to create valid props
const createTestProps = (overrides: any = {}) => ({
  id: 'test-node',
  type: 'typedListNode',
  data: {
    label: 'TypedList',
    type: 'typed_list' as const,
    typedList: {
      item_type: 'builtins.int',
      items: [1, 2, 3],
    },
    ...overrides,
  },
  selected: false,
  zIndex: 1,
  isConnectable: true,
  xPos: 0,
  yPos: 0,
  dragging: false,
});

describe('TypedListNode', () => {
  it('renders typed list node with basic information', () => {
    const props = createTestProps();
    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('TypedList')).toBeInTheDocument();
    expect(screen.getByText('builtins.int')).toBeInTheDocument();
    expect(screen.getByText('[1, 2, 3]')).toBeInTheDocument();
  });

  it('renders typed list node with result when result is provided', () => {
    const props = createTestProps({ result: [4, 5, 6] });
    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('TypedList')).toBeInTheDocument();
    expect(screen.getByText('Result:')).toBeInTheDocument();
    expect(screen.getByText('[4, 5, 6]')).toBeInTheDocument();
  });

  it('does not render result section when no result is provided', () => {
    const props = createTestProps();
    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.queryByText('Result:')).not.toBeInTheDocument();
  });

  it('displays different item types correctly', () => {
    const props = createTestProps({
      typedList: {
        item_type: 'builtins.str',
        items: ['hello', 'world'],
      },
    });

    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('builtins.str')).toBeInTheDocument();
    expect(screen.getByText('[hello, world]')).toBeInTheDocument();
  });

  it('handles custom label correctly', () => {
    const props = createTestProps({
      label: 'My Custom List',
    });

    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('My Custom List')).toBeInTheDocument();
  });

  it('defaults to TypedList when no label is provided', () => {
    const props = createTestProps({
      label: undefined,
    });

    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('TypedList')).toBeInTheDocument();
  });

  it('renders with typed list node class', () => {
    const props = createTestProps();
    const { container } = render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(container.querySelector('.typed-list-node')).toBeInTheDocument();
  });

  it('renders source and target handles', () => {
    const props = createTestProps();
    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByTestId('handle-target')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source')).toBeInTheDocument();
  });

  it('handles empty lists', () => {
    const props = createTestProps({
      typedList: {
        item_type: 'builtins.int',
        items: [],
      },
    });

    render(
      <ChakraWrapper>
        <TypedListNode {...props} />
      </ChakraWrapper>
    );

    expect(screen.getByText('builtins.int')).toBeInTheDocument();
    expect(screen.getByText('[]')).toBeInTheDocument();
  });
}); 