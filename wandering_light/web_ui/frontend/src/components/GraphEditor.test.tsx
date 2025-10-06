import React from 'react';
import { render, screen } from '@testing-library/react';
import { ChakraProvider } from '@chakra-ui/react';
import GraphEditor from './GraphEditor';

// Mock Chakra toast hook to avoid act warnings
jest.mock('@chakra-ui/react', () => {
  const actual = jest.requireActual('@chakra-ui/react');
  return { ...actual, useToast: () => jest.fn() };
});

// Mock axios
jest.mock('axios', () => ({
  __esModule: true,
  default: {
    get: jest.fn(() => Promise.resolve({ data: [] })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
  },
}));

// Mock ReactFlow components
jest.mock('reactflow', () => ({
  __esModule: true,
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <div data-testid="reactflow-provider">{children}</div>,
  default: () => <div data-testid="reactflow" />,
  addEdge: jest.fn(),
  useNodesState: () => [[], jest.fn(), jest.fn()],
  useEdgesState: () => [[], jest.fn(), jest.fn()],
  Controls: () => <div data-testid="controls" />,
  Background: () => <div data-testid="background" />,
  ConnectionLineType: {},
  Handle: ({ children, type }: any) => <div data-testid={`handle-${type}`}>{children}</div>,
  Position: { Left: 'left', Right: 'right' },
}));

// Mock Sidebar component
jest.mock('./Sidebar', () => {
  return function MockSidebar() {
    return <div data-testid="sidebar">Mocked Sidebar</div>;
  };
});

// Wrapper component for Chakra UI
const ChakraWrapper = ({ children }: { children: React.ReactNode }) => (
  <ChakraProvider>{children}</ChakraProvider>
);

describe('GraphEditor', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock WebSocket to avoid errors during unmount
    (global as any).WebSocket = jest.fn(() => ({
      send: jest.fn(),
      close: jest.fn(),
      readyState: 1,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    }));
  });

  it('renders without crashing', () => {
    render(
      <ChakraWrapper>
        <GraphEditor />
      </ChakraWrapper>
    );

    expect(screen.getByTestId('reactflow-provider')).toBeInTheDocument();
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
  });

  it('renders the execute button', () => {
    render(
      <ChakraWrapper>
        <GraphEditor />  
      </ChakraWrapper>
    );

    expect(screen.getByText('Execute Graph')).toBeInTheDocument();
  });

}); 